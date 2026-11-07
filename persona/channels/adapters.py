"""Every way a human can be reached: app, SMS, WhatsApp, phone call, email, Telegram.

Outbound adapters are thin, real HTTP/SMTP calls. Inbound webhooks live in the
API and route the sender back to their persona, so the same being answers
whichever door the person knocks on.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import smtplib
from email.message import EmailMessage
from xml.sax.saxutils import escape

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    import httpx2 as httpx  # type: ignore

from ..config import Settings
from ..store import jload

log = logging.getLogger(__name__)

CHANNELS = ("app", "sms", "whatsapp", "call", "email", "telegram")


class DeliveryError(RuntimeError):
    pass


class Channels:
    def __init__(self, settings: Settings, http: "httpx.AsyncClient | None" = None):
        self.s = settings
        self._http = http
        self.sent: list[dict] = []  # last deliveries (observability / tests)

    def http(self) -> "httpx.AsyncClient":
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30)
        return self._http

    def available(self) -> list[str]:
        out = ["app"]
        if self.s.twilio_sid and self.s.twilio_token and self.s.twilio_from:
            out += ["sms", "call"]
        if self.s.twilio_sid and self.s.twilio_token and self.s.twilio_whatsapp_from:
            out.append("whatsapp")
        if self.s.smtp_host and self.s.smtp_from:
            out.append("email")
        if self.s.telegram_token:
            out.append("telegram")
        return out

    def pick(self, persona: dict, preferred: str | None = None) -> str:
        contact = jload(persona.get("contact"), {}) or {}
        avail = self.available()
        for ch in (preferred, persona.get("preferred_channel"), "app"):
            if ch and ch in avail and (ch == "app" or contact.get("sms" if ch == "call" else ch)):
                return ch
        return "app"

    async def deliver(self, persona: dict, channel: str, text: str, *, subject: str = "") -> None:
        contact = jload(persona.get("contact"), {}) or {}
        if channel == "app":
            return  # outbox + live bus already carry it
        if channel not in self.available():
            raise DeliveryError(f"channel {channel} is not configured on this server")
        if channel in ("sms", "whatsapp"):
            to = contact.get(channel)
            sender = self.s.twilio_from if channel == "sms" else self.s.twilio_whatsapp_from
            if channel == "whatsapp":
                to = to if str(to).startswith("whatsapp:") else f"whatsapp:{to}"
                sender = sender if sender.startswith("whatsapp:") else f"whatsapp:{sender}"
            await self._twilio("Messages.json", {"To": to, "From": sender, "Body": text[:1500]})
        elif channel == "call":
            twiml = f"<Response><Say>{escape(text[:3000])}</Say></Response>"
            await self._twilio("Calls.json", {"To": contact.get("sms"), "From": self.s.twilio_from, "Twiml": twiml})
        elif channel == "email":
            await asyncio.to_thread(self._smtp, contact.get("email"), subject or f"From {persona.get('name') or 'me'}", text)
        elif channel == "telegram":
            r = await self.http().post(f"https://api.telegram.org/bot{self.s.telegram_token}/sendMessage",
                                       json={"chat_id": contact.get("telegram"), "text": text[:4000]})
            if r.status_code >= 400:
                raise DeliveryError(f"telegram {r.status_code}: {r.text[:200]}")
        else:
            raise DeliveryError(f"unknown channel {channel}")
        self.sent.append({"channel": channel, "text": text})

    async def _twilio(self, resource: str, form: dict) -> None:
        if not form.get("To"):
            raise DeliveryError("no destination number on file")
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.s.twilio_sid}/{resource}"
        r = await self.http().post(url, data=form, auth=(self.s.twilio_sid, self.s.twilio_token))
        if r.status_code >= 400:
            raise DeliveryError(f"twilio {r.status_code}: {r.text[:200]}")

    def _smtp(self, to: str | None, subject: str, body: str) -> None:
        if not to:
            raise DeliveryError("no email on file")
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.s.smtp_from, to, subject
        msg.set_content(body)
        with smtplib.SMTP(self.s.smtp_host, self.s.smtp_port, timeout=30) as smtp:
            smtp.starttls()
            if self.s.smtp_user:
                smtp.login(self.s.smtp_user, self.s.smtp_password)
            smtp.send_message(msg)


def twilio_signature_ok(auth_token: str, url: str, params: dict, signature: str) -> bool:
    """Validate X-Twilio-Signature (HMAC-SHA1 over URL + sorted params)."""
    if not auth_token:
        return False
    data = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    digest = hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature or "")
