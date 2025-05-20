#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json


def load_weights(weights_file_path=None):
    """
    بارگذاری وزن‌های توابع هدف از فایل JSON یا استفاده از مقادیر پیش‌فرض

    پارامترها:
        weights_file_path (str): مسیر فایل JSON حاوی وزن‌ها (اختیاری)

    خروجی:
        tuple: وزن‌های w1، w2، w3 و وضعیت اعتبارسنجی
    """
    # مقادیر پیش‌فرض
    default_weights = {
        "w1": 0.33,  # وزن هزینه تأمین واکسن
        "w2": 0.33,  # وزن هزینه‌های اجتماعی
        "w3": 0.34  # وزن هزینه‌های اقتصادی
    }

    weights = default_weights.copy()
    source = "پیش‌فرض"
    validation_message = ""

    # بارگذاری از فایل JSON (اگر مسیر ارائه شده باشد)
    if weights_file_path:
        if os.path.exists(weights_file_path):
            try:
                with open(weights_file_path, 'r', encoding='utf-8') as file:
                    loaded_weights = json.load(file)

                # بررسی وجود کلیدهای مورد نیاز
                required_keys = ["w1", "w2", "w3"]
                if all(key in loaded_weights for key in required_keys):
                    weights = loaded_weights
                    source = weights_file_path
                else:
                    missing_keys = [key for key in required_keys if key not in loaded_weights]
                    validation_message = f"خطا: کلیدهای {', '.join(missing_keys)} در فایل وزن‌ها یافت نشد. از مقادیر پیش‌فرض استفاده می‌شود."
            except json.JSONDecodeError:
                validation_message = f"خطا: فایل {weights_file_path} قالب JSON معتبری ندارد. از مقادیر پیش‌فرض استفاده می‌شود."
        else:
            validation_message = f"هشدار: فایل {weights_file_path} یافت نشد. از مقادیر پیش‌فرض استفاده می‌شود."

    # استخراج مقادیر
    w1 = float(weights["w1"])
    w2 = float(weights["w2"])
    w3 = float(weights["w3"])

    # اعتبارسنجی: مجموع وزن‌ها باید برابر 1 باشد
    weights_sum = w1 + w2 + w3
    is_valid = abs(weights_sum - 1.0) < 0.0001  # مقایسه با دقت کافی برای اعداد اعشاری

    if not is_valid:
        validation_message = f"خطا: مجموع وزن‌ها ({weights_sum:.4f}) برابر 1 نیست. وزن‌ها نرمالیزه می‌شوند."
        # نرمالیزه کردن وزن‌ها
        w1 = w1 / weights_sum
        w2 = w2 / weights_sum
        w3 = w3 / weights_sum

    print("\n=== اطلاعات وزن‌های توابع هدف ===")
    print(f"منبع وزن‌ها: {source}")

    if validation_message:
        print(validation_message)

    print(f"w1 (هزینه تأمین واکسن): {w1:.4f}")
    print(f"w2 (هزینه‌های اجتماعی): {w2:.4f}")
    print(f"w3 (هزینه‌های اقتصادی): {w3:.4f}")
    print(f"مجموع: {w1 + w2 + w3:.4f}")

    return w1, w2, w3, is_valid