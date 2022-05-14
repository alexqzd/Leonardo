#!/usr/bin/env python
# pylint: disable=C0116,W0613
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, ChatAction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler, ConversationHandler

from enum import Enum
import os
telegram_token = os.environ['TELEGRAM_TOKEN']
openai_api_key = os.environ['OPENAI_API_KEY']

import openai
openai.api_key = openai_api_key

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

# User and AI enum
class Entity(str, Enum):
    """Entity enum."""
    USER = "User"
    AI = "AI"

_chat_log = []

def append_to_chat_log(entity: Entity, message: str):
    """Append a message to the chat log."""
    _chat_log.append(f"{entity.value}: {message.strip()}")

def get_chat_log() -> str:
    """Get the chat log."""
    return "\n".join(_chat_log)

def get_chat_response(message: str) -> str:
    """Get the chat response."""
    append_to_chat_log(Entity.USER, message)
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt = f"{get_chat_log()}\n{Entity.USER.value}: {message}\n{Entity.AI.value}:",
        temperature=0.9,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0.6,
        presence_penalty=0.9
    )
    response = response.choices[0].text
    append_to_chat_log(Entity.AI, response)
    return response

# Convo states
Waiting_for_chat_message, Waiting_for_raw_prompt = range(2)

whitelist = [1080659616]

def authorized(user) -> bool:
    """Check if the user is authorized."""
    if user.id not in whitelist:
        print(f"User {user.full_name} ({user.id}) is not authorized")
        return False
    else:
        return True

# Define a few command handlers. These usually take the two arguments update and
# context.

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    if not authorized(update.effective_user): return
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    print(f"Message: {update.message.text.lower()}")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"You are a chatbot named Leonardo. Write a message greeting {update.effective_user.full_name}.",
        temperature=0.9,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6
    )
    response = response.choices[0].text
    update.message.reply_text(response)
    _chat_log.clear()
    append_to_chat_log(Entity.AI, response)
    return Waiting_for_chat_message

def process_message(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    if not authorized(update.effective_user): return
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    message = update.message.text
    print(f"Message: {message}")
    response = get_chat_response(message)
    update.message.reply_text(response)
    print(f"Response: {response}")
    return Waiting_for_chat_message

def process_raw_prompt(update: Update, context: CallbackContext) -> None:
    """Pass the raw prompt to the AI."""
    if not authorized(update.effective_user): return
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    message = update.message.text
    print(f"Message: {message}")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=message,
        temperature=0.9,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6
    )
    response = response.choices[0].text
    print(f"Raw response: {response}")
    update.message.reply_text(response)
    return Waiting_for_raw_prompt

def switch_to_raw_prompt(update: Update, context: CallbackContext) -> None:
    """Switch to raw prompt."""
    if not authorized(update.effective_user): return
    update.message.reply_text("I will now pass your messages directly to OpenAI. Write a message to continue.")
    return Waiting_for_raw_prompt

def switch_to_chat_mode(update: Update, context: CallbackContext) -> None:
    """Switch to chat mode.""" 
    if not authorized(update.effective_user): return
    update.message.reply_text("Chat mode is now enabled, I try to keep track of our recent conversation.")
    update.message.reply_text("If something goes wrong, you can restart the conversation by typing /start.")
    update.message.reply_text("Write a message to continue.")
    return Waiting_for_chat_message

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    if not authorized(update.effective_user): return
    update.message.reply_text('You can ask me things, try it!')
    update.message.reply_text('You can switch to raw prompt mode by typing /raw_mode')
    update.message.reply_text('You can switch to chat mode by typing /chat_mode (this is the default)')
    update.message.reply_text('You can reset the chat mode conversation by typing /start')

def process_start_message(update: Update, context: CallbackContext) -> None:
    """Reply the user message."""
    print("The user has started the conversation without the")
    if not authorized(update.effective_user): return
    update.message.reply_text('Hi! Send the /start command to start the conversation.')

mode_switch_commands_handler = [CommandHandler('chat_mode', switch_to_chat_mode), CommandHandler('raw_mode', switch_to_raw_prompt)]

def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(telegram_token)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(Filters.text & ~Filters.command, process_start_message), CommandHandler('start', start)],
        states={
            Waiting_for_chat_message: [
                MessageHandler(Filters.text & ~Filters.command, process_message),
                CommandHandler('help', help_command),CommandHandler('start', start)] + mode_switch_commands_handler,
            Waiting_for_raw_prompt: [
                MessageHandler(Filters.text & ~Filters.command,process_raw_prompt),
                CommandHandler('help', help_command)] + mode_switch_commands_handler,
        },
        fallbacks=[CommandHandler('help', help)],
    )

    dispatcher.add_handler(conv_handler)

    PORT = int(os.environ.get('PORT', '8443'))

    # Start the Bot
    updater.start_webhook(listen="0.0.0.0",
    port=int(PORT),
    url_path=telegram_token,
    webhook_url='https://leonardo-bot.herokuapp.com/' + telegram_token)
    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    print("Starting bot!")
    main()
