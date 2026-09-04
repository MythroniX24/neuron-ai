"""
Email Notification Utility for Neuron-AI
Send training status, alerts, and reports via email.

Usage:
    from continuum.utils.email_notifier import EmailNotifier
    
    # Gmail (App Password required)
    notifier = EmailNotifier(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        sender_email="your-email@gmail.com",
        sender_password="your-app-password",  # NOT your real password!
    )
    
    notifier.send(
        to="recipient@example.com",
        subject="Training Complete",
        body="Model trained successfully in 30 minutes."
    )
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional
from datetime import datetime


class EmailNotifier:
    """Send emails via SMTP. Supports Gmail, Outlook, and custom servers."""

    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.environ.get("EMAIL_SENDER", "")
        self.sender_password = sender_password or os.environ.get("EMAIL_PASSWORD", "")

    def send(
        self,
        to: str,
        subject: str,
        body: str,
        body_html: Optional[str] = None,
        attachments: Optional[list] = None,
    ) -> bool:
        """
        Send an email.
        
        Args:
            to: Recipient email address
            subject: Email subject line
            body: Plain text body
            body_html: Optional HTML body (if None, uses plain text)
            attachments: Optional list of file paths to attach
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.sender_email or not self.sender_password:
            print("⚠️  Email not configured. Set EMAIL_SENDER and EMAIL_PASSWORD env vars.")
            return False

        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = to
        msg["Subject"] = f"[Neuron-AI] {subject}"

        msg.attach(MIMEText(body, "plain"))
        if body_html:
            msg.attach(MIMEText(body_html, "html"))

        # Attach files
        if attachments:
            for filepath in attachments:
                try:
                    with open(filepath, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    filename = os.path.basename(filepath)
                    part.add_header("Content-Disposition", f"attachment; filename={filename}")
                    msg.attach(part)
                except FileNotFoundError:
                    print(f"⚠️  Attachment not found: {filepath}")

        try:
            # ⚡ FIX: add a 30s timeout — a dead/unreachable SMTP server used to
            # hang the whole training loop indefinitely.
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            print(f"✅ Email sent to {to}: {subject}")
            return True
        except smtplib.SMTPAuthenticationError:
            print("❌ Authentication failed. Check your email/password (use App Password for Gmail).")
            return False
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False

    def send_training_alert(
        self,
        to: str,
        status: str,
        epoch: int,
        total_epochs: int,
        loss: float,
        eta_minutes: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """Send a training status notification."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"""
🧠 Neuron-AI Training Update
{'='*40}
Status:   {status}
Epoch:    {epoch}/{total_epochs}
Loss:     {loss:.6f}
Time:     {timestamp}
"""
        if eta_minutes is not None:
            body += f"ETA:      {eta_minutes:.0f} minutes\n"
        if checkpoint_path:
            body += f"Checkpoint: {checkpoint_path}\n"

        body += f"\n{'='*40}\nPowered by Neuron-AI 🚀"
        
        subject = f"Training {status} — Epoch {epoch}/{total_epochs} (Loss: {loss:.4f})"
        attachments = [checkpoint_path] if checkpoint_path else None
        return self.send(to, subject, body, attachments=attachments)


# Convenience: create notifier from environment
def get_notifier() -> EmailNotifier:
    """Create EmailNotifier from environment variables."""
    return EmailNotifier(
        smtp_server=os.environ.get("SMTP_SERVER", "smtp.gmail.com"),
        smtp_port=int(os.environ.get("SMTP_PORT", "587")),
    )


if __name__ == "__main__":
    # Test
    notifier = get_notifier()
    notifier.send(
        to="test@example.com",
        subject="Test Email",
        body="This is a test from Neuron-AI email notifier.",
    )
