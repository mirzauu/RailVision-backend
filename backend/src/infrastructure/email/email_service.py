from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from pydantic import EmailStr
from src.config.settings import settings
from typing import List, Optional
import os

class EmailService:
    def __init__(self):
        self.config = ConnectionConfig(
            MAIL_USERNAME=settings.mail_username,
            MAIL_PASSWORD=settings.mail_password,
            MAIL_FROM=settings.mail_from,
            MAIL_PORT=settings.mail_port,
            MAIL_SERVER=settings.mail_server,
            MAIL_FROM_NAME=settings.mail_from_name,
            MAIL_STARTTLS=settings.mail_starttls,
            MAIL_SSL_TLS=settings.mail_ssl_tls,
            USE_CREDENTIALS=settings.use_credentials,
            VALIDATE_CERTS=settings.validate_certs,
            # TEMPLATE_FOLDER=os.path.join(os.path.dirname(__file__), "templates")
        )
        self.fastmail = FastMail(self.config)

    async def send_email(self, subject: str, recipients: List[str], body: str, subtype: MessageType = MessageType.html):
        message = MessageSchema(
            subject=subject,
            recipients=recipients,
            body=body,
            subtype=subtype
        )
        await self.fastmail.send_message(message)

    async def send_otp(self, email: str, otp: str):
        subject = "Your Reset Password OTP"
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                    <h2 style="color: #2c3e50; text-align: center;">Reset Your Password</h2>
                    <p>Hello,</p>
                    <p>You requested to reset your password. Please use the Following One-Time Password (OTP) to complete the process:</p>
                    <div style="text-align: center; margin: 30px 0;">
                        <span style="font-size: 32px; font-weight: bold; letter-spacing: 5px; color: #e74c3c; background: #f9f9f9; padding: 10px 20px; border-radius: 5px;">{otp}</span>
                    </div>
                    <p>This OTP is valid for 10 minutes. If you did not request a password reset, please ignore this email.</p>
                    <p>Best regards,<br>The RailVision Team</p>
                </div>
            </body>
        </html>
        """
        await self.send_email(subject, [email], body)

    async def send_invitation(self, email: str, invitation_link: str, inviter_name: str, org_name: str):
        subject = f"Invitation to join {org_name} on RailVision"
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                    <h2 style="color: #2c3e50; text-align: center;">You're Invited!</h2>
                    <p>Hello,</p>
                    <p><strong>{inviter_name}</strong> has invited you to join <strong>{org_name}</strong> on RailVision.</p>
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{invitation_link}" style="background-color: #3498db; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold;">Accept Invitation</a>
                    </div>
                    <p>If the button doesn't work, you can copy and paste the following link into your browser:</p>
                    <p>{invitation_link}</p>
                    <p>Best regards,<br>The RailVision Team</p>
                </div>
            </body>
        </html>
        """
        await self.send_email(subject, [email], body)
