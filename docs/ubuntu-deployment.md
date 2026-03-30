# Ubuntu Deployment

This project does not need `screen` or `tmux` to run on Ubuntu. For production use, run the scheduler as a `systemd` service.

## Assumptions

- Repo path: `/opt/insider_alert`
- Service user: `ubuntu`
- Python version: `3.10+`

Adjust paths and user names to match your server.

## Initial setup

```bash
sudo mkdir -p /opt/insider_alert
sudo chown "$USER":"$USER" /opt/insider_alert
git clone <YOUR_REPO_URL> /opt/insider_alert
cd /opt/insider_alert
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and `config.yaml` before starting the service.

## systemd service

Copy the example service file from [`deploy/systemd/insider-alert.service`](/opt/insider_alert/deploy/systemd/insider-alert.service) to `/etc/systemd/system/insider-alert.service` and adjust `User` and `WorkingDirectory` if needed.

Then enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable insider-alert
sudo systemctl start insider-alert
```

## Operations

Check service status:

```bash
sudo systemctl status insider-alert
```

Follow logs:

```bash
journalctl -u insider-alert -f
```

Restart after config or code changes:

```bash
sudo systemctl restart insider-alert
```

Stop the scheduler:

```bash
sudo systemctl stop insider-alert
```

## Deploying updates

Repo changes do not affect an already running scheduler process until it is restarted.

Recommended update flow:

```bash
cd /opt/insider_alert
git pull
./.venv/bin/pip install -r requirements.txt
sudo systemctl restart insider-alert
sudo systemctl status insider-alert
```

If `requirements.txt` did not change, the pip step can be skipped.

## Notes

- `python main.py scan` is a one-off run and exits when finished.
- `python main.py schedule` starts a long-running blocking scheduler.
- SQLite data is written to `insider_alert.db` in the project directory unless you change the code or runtime environment.
