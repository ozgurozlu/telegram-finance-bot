import os
import telebot
from analyze import analyze_asset

# Telegram Bot Token (Render'da Environment Variable olarak TELEGRAM_TOKEN tanımlanmalı)
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN environment variable is not set!")

bot = telebot.TeleBot(TOKEN)

# /start komutu
@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(
        message,
        "Merhaba! 📊\n"
        "Bana şu şekilde yaz:\n"
        "`/tahmin BTC-USD 7`\n"
        "- İlk kısım: Sembol (örnek: BTC-USD, AAPL)\n"
        "- İkinci kısım: Gün sayısı (örnek: 7, 30, 90)",
        parse_mode="Markdown"
    )

# /tahmin komutu
@bot.message_handler(commands=["tahmin"])
def handle_analysis(message):
    try:
        parts = message.text.split()
        if len(parts) != 3:
            bot.reply_to(message, "❌ Format yanlış! Örnek: `/tahmin BTC-USD 7`", parse_mode="Markdown")
            return
        
        symbol = parts[1]
        days = int(parts[2])

        bot.send_message(message.chat.id, f"🔍 {symbol} için analiz yapılıyor...")

        report, chart_path = analyze_asset(symbol, days)

        # Yazılı rapor gönder
        bot.send_message(message.chat.id, f"```\n{report}\n```", parse_mode="Markdown")

        # Grafik gönder
        with open(chart_path, "rb") as chart:
            bot.send_photo(message.chat.id, chart)

    except Exception as e:
        bot.reply_to(message, f"❌ Hata: {str(e)}")

# Botu çalıştır
if __name__ == "__main__":
    print("Bot çalışıyor...")
    bot.polling(none_stop=True)
