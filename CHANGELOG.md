# Changelog

All notable changes to insider_alert will be documented in this file.
Format: Phases from the upgrade plan + ad-hoc changes.

Legend:
- 🗄️ **DB** — New/modified database tables (auto-created on restart)
- 📦 **Deps** — New/changed dependencies → run `pip install -r requirements.txt`
- ⚙️ **Config** — Changes to `config.yaml` format or defaults
- 🔄 **Restart** — Service restart required (`sudo systemctl restart insider-alert`)

---

## [Unreleased]

### Pre-Phase Cleanup
- Created `.github/copilot-instructions.md` for VS Code Copilot context
- Created `deploy/upgrade.sh` for automated server upgrades
- Created this CHANGELOG.md

---

<!-- Template for future entries:

## Phase XX — Title (YYYY-MM-DD)

### Added
- ...

### Changed
- ...

### Migration Notes
- 📦 **Deps**: `pip install -r requirements.txt` (added: xxx)
- 🗄️ **DB**: New table `xxx` (auto-created on restart)
- ⚙️ **Config**: New key `xxx:` in config.yaml
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

-->
