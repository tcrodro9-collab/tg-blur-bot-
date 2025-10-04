import os
import telebot
from rembg import remove
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from io import BytesIO

# ---------- CONFIG ----------
TOKEN = os.getenv("BOT_TOKEN") or "8375612747:AAHcGv1YneDzAkl82OGAk4FDswRTA1JefWI"
bot = telebot.TeleBot(TOKEN)
photo_cache = {}

# ---------- HELPER FUNCTIONS ----------

def enhance_colors_limited(img):
    """Boost colors, limited green/blue saturation"""
    cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    # Green (35–85), Blue (85–135) → limited boost only
    green_mask = (hue > 35) & (hue < 85)
    blue_mask = (hue > 85) & (hue < 135)

    sat[green_mask] *= 1.12  # only 12% boost
    sat[blue_mask] *= 1.10   # only 10% boost

    # global saturation slight boost
    sat *= 1.08
    sat = np.clip(sat, 0, 255)
    hsv[:, :, 1] = sat

    # Back to BGR
    cv_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Slight warm tone
    b, g, r = cv_img[:, :, 0], cv_img[:, :, 1], cv_img[:, :, 2]
    r = cv2.add(r, 10)
    b = cv2.subtract(b, 5)
    warm_img = cv2.merge((b, g, r))

    return Image.fromarray(cv2.cvtColor(warm_img, cv2.COLOR_BGR2RGB))


def soften_subject(subject_img, mask):
    """Lighten and smooth subject"""
    subject = np.array(subject_img.convert("RGB"), dtype=np.float32)
    alpha = np.array(mask.resize(subject_img.size)).astype(np.float32) / 255.0

    # Slight brighten only inside subject
    brighten = subject * 1.07 + 6
    brighten = np.clip(brighten, 0, 255)

    # Smooth using bilateral filter
    blur = cv2.bilateralFilter(brighten.astype(np.uint8), d=5, sigmaColor=50, sigmaSpace=50)

    result = (subject * (1 - alpha[..., None]) + blur * alpha[..., None]).astype(np.uint8)
    return Image.fromarray(result.astype(np.uint8))


def create_bokeh(size, density=160, color_tone=(255, 220, 180)):
    h, w = size
    bokeh = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(density):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(8, 35)
        color = (
            np.clip(color_tone[0] + np.random.randint(-30, 30), 0, 255),
            np.clip(color_tone[1] + np.random.randint(-30, 30), 0, 255),
            np.clip(color_tone[2] + np.random.randint(-30, 30), 0, 255)
        )
        cv2.circle(bokeh, (x, y), radius, color, -1)
    bokeh = cv2.GaussianBlur(bokeh, (0, 0), sigmaX=25)
    return bokeh


def create_depth_blur(img, mask, blur_strength):
    """Depth blur + bokeh"""
    cv_img = cv2.cvtColor(np.array(img.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
    cv_mask = np.array(mask.resize(img.size)).astype(np.float32) / 255.0
    inv_mask = 1 - cv_mask

    h, w = inv_mask.shape
    gradient = np.tile(np.linspace(0, 1, h).reshape(h, 1), (1, w))
    depth_map = np.clip(inv_mask * 0.8 + gradient * 0.3, 0, 1)

    blur_base = cv2.GaussianBlur(cv_img, (0, 0), sigmaX=blur_strength, sigmaY=blur_strength)
    blurred = (cv_img * (1 - depth_map[..., None]) + blur_base * depth_map[..., None]).astype(np.uint8)

    # Add Bokeh lights
    bokeh_layer = create_bokeh((h, w), density=180)
    blended_bg = cv2.addWeighted(blurred[:, :, :3], 0.85, bokeh_layer, 0.4, 0)

    final_blur = Image.fromarray(cv2.cvtColor(blended_bg, cv2.COLOR_BGR2RGB)).convert("RGBA")
    return final_blur


def dslr_blur_effect(image_bytes, level):
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    blur_values = {"low": 20, "medium": 40, "high": 65}
    blur_strength = blur_values.get(level, 40)

    # Subject mask
    no_bg_bytes = remove(image_bytes)
    no_bg = Image.open(BytesIO(no_bg_bytes)).convert("RGBA")
    mask = no_bg.split()[-1]

    # Depth blur + bokeh
    blurred_img = create_depth_blur(img, mask, blur_strength)

    # Color enhancement (limited green/blue saturation)
    color_boosted = enhance_colors_limited(blurred_img)

    # Soften subject / skin
    subject_soft = soften_subject(img, mask)

    # Merge
    final = Image.composite(subject_soft, color_boosted.convert("RGBA"), mask)

    buf = BytesIO()
    final.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ---------- TELEGRAM BOT ----------

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded = bot.download_file(file_info.file_path)
    photo_cache[message.chat.id] = downloaded

    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    markup.row("Low Blur", "Medium Blur", "High Blur")
    bot.reply_to(message, "🎚 Choose DSLR blur strength:", reply_markup=markup)


@bot.message_handler(func=lambda m: m.text in ["Low Blur", "Medium Blur", "High Blur"])
def process_blur(message):
    chat_id = message.chat.id
    if chat_id not in photo_cache:
        bot.reply_to(message, "📷 Please send a photo first.")
        return

    level = message.text.lower().split()[0]
    msg = bot.send_message(chat_id, "📸 Processing DSLR blur, bokeh & color tone... ⏳")

    try:
        out_buf = dslr_blur_effect(photo_cache[chat_id], level)
        bot.send_photo(chat_id, out_buf, caption=f"✨ DSLR {level.capitalize()} Blur + Limited Green/Blue Saturation + Bokeh Ready!")
    except Exception as e:
        bot.send_message(chat_id, f"❌ Error: {e}")
    finally:
        photo_cache.pop(chat_id, None)
        bot.send_message(chat_id, "✅ Done!", reply_markup=telebot.types.ReplyKeyboardRemove())


if __name__ == "__main__":
    bot.polling(none_stop=True)
