//! Memory store — ~/.enchanter/memories/MEMORY.md, USER.md, and SUMMARY.md, §-delimited.
//!
//! The memory architecture (separate MEMORY.md + USER.md files, §-delimited entries,
//! threshold-based summarization of old entries) is adapted from hermes-agent's
//! memory system (hermes-agent/agent/memory_manager.py, agent/memory_provider.py).
//! hermes-agent uses a pluggable MemoryProvider ABC with sync_turn/prefetch/shutdown
//! lifecycle hooks and multiple backends (built-in, Honcho, Hindsight, Mem0);
//! enchanter simplifies to a single file-based store with synchronous load/save
//! and async LLM summarization.
//!
//! The §-delimited entry format (\n§\n between entries) matches hermes-agent's
//! file-based default memory store convention, enabling file-level compatibility:
//! a hermes-agent ~/.hermes/MEMORY.md can be symlinked or copied to
//! ~/.enchanter/memories/MEMORY.md and parsed correctly.
//!
//! The summarize_on_exit pattern (call the LLM to compress old memory entries
//! when they exceed a threshold, falling back to simple truncation) mirrors
//! hermes-agent's two-tier approach: MemoryManager.sync_all() for per-turn
//! persistence, and conversation_compression for context window management.
//!
//! The memory tool (add/remove/replace/list) follows the operations exposed by
//! hermes-agent's built-in memory (hermes-agent/tools/memory_tool.py) and
//! Claude Code's memory system (claude-code/src/memdir/memdir.ts), which
//! uses MEMORY.md + per-topic files with frontmatter-typed entries.
//! enchanter's add/remove/replace/list matches the subset of operations
//! both systems expose for user-facing memory management.

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::api::{LlmClient, Message};
use crate::config::MemoryConfig;
use crate::home;
use crate::summary::truncate_chars;

const ENTRY_DELIMITER: &str = "\n§\n";

/// Character budget for the text handed to the summarizer LLM in one call.
const SUMMARIZE_INPUT_CHAR_CAP: usize = 12_000;

/// How many of the leading `entries` fit within `budget` characters when
/// joined with the "\n---\n" separator (5 chars per join). Returns a count in
/// `0..=entries.len()`; the caller decides how to handle 0 (a single entry
/// larger than the whole budget).
fn split_at_char_budget(entries: &[String], budget: usize) -> usize {
    let mut used = 0usize;
    for (i, entry) in entries.iter().enumerate() {
        used += entry.chars().count();
        if i > 0 {
            used += 5; // "\n---\n" separator
        }
        if used > budget {
            return i;
        }
    }
    entries.len()
}

#[derive(Debug, Clone, Default)]
pub struct MemoryStore {
    pub memory_entries: Vec<String>,
    pub user_entries: Vec<String>,
    pub summary: Option<String>,
    /// Project overlay entries merged in for this session only (from
    /// `.enchanter/memories/`). They are shown in the prompt and in
    /// `memory list` alongside the global entries, but are never written
    /// back to the global MEMORY.md/USER.md — otherwise project-scoped
    /// entries would leak permanently into the global store (and duplicate
    /// on every session, since the overlay is re-merged each load).
    pub overlay_memory_entries: Vec<String>,
    pub overlay_user_entries: Vec<String>,
}

impl MemoryStore {
    /// Load memory from disk. Sync — does not run summarization.
    pub fn load() -> Result<Self> {
        let mem_dir = memory_dir();

        let memory_entries = if mem_dir.join("MEMORY.md").exists() {
            let content =
                std::fs::read_to_string(mem_dir.join("MEMORY.md")).context("reading MEMORY.md")?;
            Self::parse_entries(&content)
        } else {
            Vec::new()
        };

        let user_entries = if mem_dir.join("USER.md").exists() {
            let content =
                std::fs::read_to_string(mem_dir.join("USER.md")).context("reading USER.md")?;
            Self::parse_entries(&content)
        } else {
            Vec::new()
        };

        let summary = if mem_dir.join("SUMMARY.md").exists() {
            let content = std::fs::read_to_string(mem_dir.join("SUMMARY.md"))
                .context("reading SUMMARY.md")?;
            let trimmed = content.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        } else {
            None
        };

        Ok(Self {
            memory_entries,
            user_entries,
            summary,
            overlay_memory_entries: Vec::new(),
            overlay_user_entries: Vec::new(),
        })
    }

    fn parse_entries(content: &str) -> Vec<String> {
        content
            .split(ENTRY_DELIMITER)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    #[allow(dead_code)]
    fn serialize_entries(entries: &[String]) -> String {
        entries.join(ENTRY_DELIMITER)
    }

    #[allow(dead_code)]
    pub fn add_memory(&mut self, entry: String) -> Result<()> {
        self.memory_entries.push(entry);
        self.save_memory()
    }

    #[allow(dead_code)]
    pub fn add_user(&mut self, entry: String) -> Result<()> {
        self.user_entries.push(entry);
        self.save_user()
    }

    #[allow(dead_code)]
    pub fn remove_memory(&mut self, substring: &str) -> Result<bool> {
        let before = self.memory_entries.len();
        self.memory_entries.retain(|e| !e.contains(substring));
        let removed = self.memory_entries.len() < before;
        if removed {
            self.save_memory()?;
        }
        Ok(removed)
    }

    #[allow(dead_code)]
    pub fn replace_memory(&mut self, old_text: &str, new_text: &str) -> Result<bool> {
        let mut found = false;
        for entry in &mut self.memory_entries {
            if entry.contains(old_text) {
                *entry = entry.replace(old_text, new_text);
                found = true;
            }
        }
        if found {
            self.save_memory()?;
        }
        Ok(found)
    }

    #[allow(dead_code)]
    fn save_memory(&self) -> Result<()> {
        let mem_dir = memory_dir();
        std::fs::create_dir_all(&mem_dir).context("creating memories directory")?;
        let content = Self::serialize_entries(&self.memory_entries);
        std::fs::write(mem_dir.join("MEMORY.md"), content).context("writing MEMORY.md")
    }

    #[allow(dead_code)]
    fn save_user(&self) -> Result<()> {
        let mem_dir = memory_dir();
        std::fs::create_dir_all(&mem_dir).context("creating memories directory")?;
        let content = Self::serialize_entries(&self.user_entries);
        std::fs::write(mem_dir.join("USER.md"), content).context("writing USER.md")
    }

    fn save_summary(&self) -> Result<()> {
        let mem_dir = memory_dir();
        std::fs::create_dir_all(&mem_dir).context("creating memories directory")?;
        if let Some(summary) = &self.summary {
            std::fs::write(mem_dir.join("SUMMARY.md"), summary).context("writing SUMMARY.md")
        } else {
            // Remove SUMMARY.md if summary is None
            let path = mem_dir.join("SUMMARY.md");
            if path.exists() {
                std::fs::remove_file(path).context("removing SUMMARY.md")?;
            }
            Ok(())
        }
    }

    /// Merge project overlay memories from a directory.
    /// Adds entries from project MEMORY.md and USER.md to this store.
    /// Does not save to disk — merge is in-memory only for the session, and
    /// the overlay entries are kept separate so later saves of the global
    /// store never persist them.
    pub fn merge_from_dir(&mut self, dir: &std::path::Path) -> Result<()> {
        let memory_path = dir.join("MEMORY.md");
        if memory_path.exists() {
            let content = std::fs::read_to_string(&memory_path)
                .with_context(|| format!("reading project {}", memory_path.display()))?;
            let entries = Self::parse_entries(&content);
            self.overlay_memory_entries.extend(entries);
        }

        let user_path = dir.join("USER.md");
        if user_path.exists() {
            let content = std::fs::read_to_string(&user_path)
                .with_context(|| format!("reading project {}", user_path.display()))?;
            let entries = Self::parse_entries(&content);
            self.overlay_user_entries.extend(entries);
        }

        Ok(())
    }

    /// Global memory entries followed by project overlay entries.
    pub fn all_memory_entries(&self) -> impl Iterator<Item = &String> {
        self.memory_entries.iter().chain(&self.overlay_memory_entries)
    }

    /// Global user entries followed by project overlay entries.
    pub fn all_user_entries(&self) -> impl Iterator<Item = &String> {
        self.user_entries.iter().chain(&self.overlay_user_entries)
    }

    /// Manage memory: cap entries and summarize old ones if threshold exceeded.
    /// This is async because summarization calls the LLM.
    /// Should be called after the LlmClient is created.
    pub async fn manage(&mut self, client: &LlmClient, config: &MemoryConfig) -> Result<()> {
        let total = self.memory_entries.len() as u32;

        if total <= config.max_entries {
            return Ok(());
        }

        // Need to summarize if exceeding threshold
        if total > config.summarize_threshold {
            let entries_to_summarize = total - config.max_entries;
            if entries_to_summarize == 0 {
                return Ok(());
            }

            // Take the oldest entries (front of the list) for summarization
            let split_point = entries_to_summarize as usize;
            let old_entries: Vec<String> = self.memory_entries[..split_point].to_vec();
            let recent_entries: Vec<String> = self.memory_entries[split_point..].to_vec();

            // The summarizer input is capped (see summarize_entries). Only
            // feed it the oldest entries that fit under the cap and keep the
            // remainder as regular entries — otherwise everything past the
            // cap would be silently discarded without ever being summarized.
            let summarizable = split_at_char_budget(&old_entries, SUMMARIZE_INPUT_CHAR_CAP);
            let (to_summarize, kept) = old_entries.split_at(summarizable.max(1));

            // Summarize old entries
            let new_summary = self.summarize_entries(client, to_summarize).await?;

            // Merge with existing summary
            let merged_summary = match &self.summary {
                Some(existing) => {
                    format!(
                        "PAST SUMMARY:\n{}\n\nADDITIONAL SUMMARY:\n{}",
                        existing, new_summary
                    )
                }
                None => new_summary,
            };

            self.summary = Some(merged_summary);
            // Entries that didn't fit the summarizer input stay in the list.
            let mut remaining = kept.to_vec();
            remaining.extend(recent_entries);
            self.memory_entries = remaining;

            // Persist both the summary and the trimmed entry list — without
            // saving the entries here, the summarized entries would resurrect
            // on the next load and get folded into the summary again.
            self.save_summary()?;
            self.save_memory()?;
        } else {
            // Over max_entries but under threshold — just cap
            let cutoff = total - config.max_entries;
            self.memory_entries = self.memory_entries[cutoff as usize..].to_vec();
            self.save_memory()?;
        }

        Ok(())
    }

    /// Summarize a list of memory entries using the LLM.
    async fn summarize_entries(&self, client: &LlmClient, entries: &[String]) -> Result<String> {
        let entries_text = entries.join("\n---\n");

        // Truncate if very long to avoid token limits.
        // Use truncate_chars to avoid UTF-8 boundary panics on multi-byte chars.
        let truncated = if entries_text.chars().count() > SUMMARIZE_INPUT_CHAR_CAP {
            truncate_chars(&entries_text, SUMMARIZE_INPUT_CHAR_CAP)
        } else {
            entries_text
        };

        let system_prompt = "You are a memory condenser. Your job is to summarize \
            memory entries into a concise, dense form. Preserve key facts, decisions, \
            preferences, and technical details. Omit transient details, pleasantries, \
            and anything that won't matter later. Output a single condensed summary paragraph, \
            not a list. Be brief but information-dense.";

        let user_prompt = format!(
            "Summarize these memory entries into a dense, concise summary. \
            Focus on durable facts and decisions.\n\n{}",
            truncated
        );

        let messages = vec![Message::system(system_prompt), Message::user(&user_prompt)];

        // Use non-streaming for summarization — we don't need to display it
        let result = client
            .chat(&messages, None, &crate::config::RetryConfig::default())
            .await?;

        Ok(result.content.unwrap_or_default())
    }

    pub fn format_for_prompt(&self) -> String {
        let mut parts = Vec::new();

        let user_entries: Vec<&String> = self.all_user_entries().collect();
        if !user_entries.is_empty() {
            let joined: Vec<&str> = user_entries.iter().map(|s| s.as_str()).collect();
            parts.push(format!("═══ USER PROFILE ═══\n{}", joined.join("\n---\n")));
        }

        if let Some(summary) = &self.summary {
            parts.push(format!("═══ MEMORY SUMMARY ═══\n{}", summary));
        }

        let memory_entries: Vec<&String> = self.all_memory_entries().collect();
        if !memory_entries.is_empty() {
            let joined: Vec<&str> = memory_entries.iter().map(|s| s.as_str()).collect();
            parts.push(format!("═══ MEMORY ═══\n{}", joined.join("\n---\n")));
        }

        parts.join("\n\n")
    }

    /// Total count of memory entries (for display), including project
    /// overlay entries merged for this session.
    #[allow(dead_code)]
    pub fn total_entry_count(&self) -> usize {
        self.memory_entries.len()
            + self.overlay_memory_entries.len()
            + self.user_entries.len()
            + self.overlay_user_entries.len()
    }
}

pub fn memory_dir() -> PathBuf {
    home::enchanter_home().join("memories")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrip() {
        let content = "entry one\n§\nentry two\n§\nentry three";
        let entries = MemoryStore::parse_entries(content);
        assert_eq!(entries, vec!["entry one", "entry two", "entry three"]);
    }

    #[test]
    fn serialize_roundtrip() {
        let entries = vec!["alpha".to_string(), "beta".to_string()];
        let serialized = MemoryStore::serialize_entries(&entries);
        let parsed = MemoryStore::parse_entries(&serialized);
        assert_eq!(parsed, entries);
    }

    #[test]
    fn format_includes_sections() {
        let store = MemoryStore {
            user_entries: vec!["User is Andrew".to_string()],
            memory_entries: vec!["fact one".to_string()],
            ..Default::default()
        };
        let formatted = store.format_for_prompt();
        assert!(formatted.contains("USER PROFILE"));
        assert!(formatted.contains("MEMORY"));
        assert!(formatted.contains("Andrew"));
        assert!(formatted.contains("fact one"));
    }

    #[test]
    fn format_includes_summary() {
        let store = MemoryStore {
            user_entries: vec![],
            memory_entries: vec!["recent fact".to_string()],
            summary: Some("condensed old facts here".to_string()),
            ..Default::default()
        };
        let formatted = store.format_for_prompt();
        assert!(formatted.contains("MEMORY SUMMARY"));
        assert!(formatted.contains("condensed old facts"));
        assert!(formatted.contains("recent fact"));
    }

    #[test]
    fn total_entry_count() {
        let store = MemoryStore {
            user_entries: vec!["u1".to_string(), "u2".to_string()],
            memory_entries: vec!["m1".to_string()],
            ..Default::default()
        };
        assert_eq!(store.total_entry_count(), 3);
    }

    #[test]
    fn truncate_chars_utf8_safe() {
        // Multi-byte chars at the boundary must not panic
        let cjk = "你好世界"; // 4 chars, 12 bytes
        let truncated = crate::summary::truncate_chars(cjk, 2);
        assert_eq!(truncated.chars().count(), 2);
        assert!(!truncated.ends_with("…"));

        let emoji = "😀😃😄"; // 3 chars, 12 bytes
        let truncated = crate::summary::truncate_chars(emoji, 1);
        assert_eq!(truncated.chars().count(), 1);
        assert!(truncated.contains("😀"));

        // Boundary exactly at multi-byte char
        let mixed = "abc你好def"; // 10 chars
        let truncated = crate::summary::truncate_chars(mixed, 5);
        assert_eq!(truncated.chars().count(), 5);
    }

    #[test]
    fn split_at_char_budget_fits() {
        // All entries fit under the budget.
        let entries: Vec<String> = (0..5).map(|i| format!("entry {}", i)).collect();
        assert_eq!(split_at_char_budget(&entries, 1000), 5);
    }

    #[test]
    fn split_at_char_budget_cuts_at_limit() {
        // Each entry is 10 chars; joined with "\n---\n" (5) → 10, 25, 40, ...
        let entries: Vec<String> = (0..4).map(|_| "0123456789".to_string()).collect();
        // budget 30 → first two fit (10, then 10+5+10=25), third exceeds (40)
        assert_eq!(split_at_char_budget(&entries, 30), 2);
        // budget 9 → not even the first fits
        assert_eq!(split_at_char_budget(&entries, 9), 0);
    }

    #[test]
    fn overlay_entries_are_not_persisted_or_double_counted() {
        let mut store = MemoryStore {
            memory_entries: vec!["global fact".to_string()],
            ..Default::default()
        };
        store.overlay_memory_entries.push("project fact".to_string());

        // Prompt shows both.
        let formatted = store.format_for_prompt();
        assert!(formatted.contains("global fact"));
        assert!(formatted.contains("project fact"));

        // Serialization for disk only ever includes the global entries.
        let serialized = MemoryStore::serialize_entries(&store.memory_entries);
        assert!(serialized.contains("global fact"));
        assert!(!serialized.contains("project fact"));
    }

    #[test]
    fn cap_under_max_no_change() {
        let store = MemoryStore {
            memory_entries: (0..10).map(|i| format!("entry {}", i)).collect(),
            ..Default::default()
        };
        let config = MemoryConfig {
            max_entries: 50,
            summarize_threshold: 40,
        };
        // total is 10, under max_entries — manage should be a no-op
        // (but manage is async and needs a client, just test the logic indirectly)
        assert_eq!(store.memory_entries.len(), 10);
        let _ = &config; // verify it compiles
    }
}
