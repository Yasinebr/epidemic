#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os


def load_weights(weights_file_path=None):
    """
    بارگذاری وزن‌های تابع هدف از فایل JSON یا مقادیر پیش‌فرض

    پارامترها:
        weights_file_path (str): مسیر فایل وزن‌ها (اختیاری)

    خروجی:
        w1, w2, w3: وزن‌های توابع هدف
        weights_valid: آیا وزن‌ها معتبر هستند؟
    """
    # مقادیر پیش‌فرض
    w1, w2, w3 = 0.33, 0.33, 0.34
    weights_valid = True
    weight_name = "وزن متعادل (پیش‌فرض)"

    # اگر فایل وزن‌ها ارائه شده باشد، آن را بارگذاری می‌کنیم
    if weights_file_path and os.path.exists(weights_file_path):
        try:
            with open(weights_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # بررسی می‌کنیم آیا فایل شامل وزن‌های مورد نیاز است
                if isinstance(data, dict) and all(key in data for key in ['w1', 'w2', 'w3']):
                    w1 = data['w1']
                    w2 = data['w2']
                    w3 = data['w3']
                    # اگر نام وزن در فایل وجود داشته باشد، آن را استخراج می‌کنیم
                    weight_name = data.get('name', f"وزن سفارشی ({w1}, {w2}, {w3})")
                else:
                    # اگر فایل ساختار استاندارد نداشته باشد، از مقادیر پیش‌فرض استفاده می‌کنیم
                    print("هشدار: ساختار فایل وزن‌ها معتبر نیست.")
                    print("از مقادیر پیش‌فرض استفاده می‌شود.")
                    weights_valid = False

            # بررسی می‌کنیم که مجموع وزن‌ها تقریباً برابر با 1 باشد
            if not 0.99 <= w1 + w2 + w3 <= 1.01:
                print("هشدار: مجموع وزن‌ها باید تقریباً برابر با 1 باشد.")
                print("از مقادیر پیش‌فرض استفاده می‌شود.")
                w1, w2, w3 = 0.33, 0.33, 0.34
                weight_name = "وزن متعادل (پیش‌فرض)"
                weights_valid = False

            # بررسی می‌کنیم که وزن‌ها بین 0 و 1 باشند
            if not (0 <= w1 <= 1 and 0 <= w2 <= 1 and 0 <= w3 <= 1):
                print("هشدار: وزن‌ها باید بین 0 و 1 باشند.")
                print("از مقادیر پیش‌فرض استفاده می‌شود.")
                w1, w2, w3 = 0.33, 0.33, 0.34
                weight_name = "وزن متعادل (پیش‌فرض)"
                weights_valid = False

            print(f"وزن‌های بارگذاری شده: {weight_name} (w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f})")

        except json.JSONDecodeError:
            print("خطا در خواندن فایل JSON. از مقادیر پیش‌فرض استفاده می‌شود.")
            weights_valid = False
        except Exception as e:
            print(f"خطا در بارگذاری وزن‌ها: {e}")
            print("از مقادیر پیش‌فرض استفاده می‌شود.")
            weights_valid = False
    else:
        print(f"از مقادیر پیش‌فرض استفاده می‌شود: w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}")

    return w1, w2, w3, weights_valid, weight_name


def is_duplicate_weight(weight, weight_list):
    """
    بررسی اینکه آیا وزنی تکراری است یا خیر

    پارامترها:
        weight (dict): وزن برای بررسی
        weight_list (list): لیست وزن‌های موجود

    خروجی:
        bool: True اگر تکراری باشد، False در غیر این صورت
    """
    for w in weight_list:
        if (abs(w['w1'] - weight['w1']) < 0.01 and
                abs(w['w2'] - weight['w2']) < 0.01 and
                abs(w['w3'] - weight['w3']) < 0.01):
            return True
    return False


def load_multiple_weights(weights_file_path=None):
    """
    بارگذاری مجموعه‌ای از وزن‌ها برای اجرای چندگانه

    پارامترها:
        weights_file_path (str): مسیر فایل وزن‌ها (اختیاری)

    خروجی:
        weight_sets: لیستی از وزن‌های مختلف
    """
    # مجموعه وزن‌های پیش‌فرض
    default_weight_sets = [
        {'w1': 0.33, 'w2': 0.33, 'w3': 0.34, 'name': 'وزن متعادل'},
        {'w1': 0.8, 'w2': 0.1, 'w3': 0.1, 'name': 'تأکید شدید بر هزینه تأمین'},
        {'w1': 0.1, 'w2': 0.8, 'w3': 0.1, 'name': 'تأکید شدید بر هزینه اجتماعی'},
        {'w1': 0.1, 'w2': 0.1, 'w3': 0.8, 'name': 'تأکید شدید بر هزینه اقتصادی'}
    ]

    # اگر فایل وزن‌ها ارائه شده باشد، آن را بارگذاری می‌کنیم
    if weights_file_path and os.path.exists(weights_file_path):
        try:
            with open(weights_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # بررسی ساختار فایل
                if isinstance(data, dict) and all(key in data for key in ['w1', 'w2', 'w3']):
                    # اگر فایل فقط شامل یک مجموعه وزن باشد، آن را به فرمت لیست تبدیل می‌کنیم
                    weight_name = data.get('name', f"وزن سفارشی ({data['w1']}, {data['w2']}, {data['w3']})")
                    custom_weight = {'w1': data['w1'], 'w2': data['w2'], 'w3': data['w3'], 'name': weight_name}

                    # شروع با وزن سفارشی کاربر
                    weight_sets = [custom_weight]
                    print(f"وزن سفارشی بارگذاری شد: {weight_name}")

                    # افزودن وزن‌های پیش‌فرض بدون تکرار
                    added_count = 0
                    for ws in default_weight_sets:
                        if not is_duplicate_weight(ws, weight_sets):
                            weight_sets.append(ws)
                            added_count += 1
                        else:
                            print(f"وزن تکراری حذف شد: {ws['name']}")

                    print(f"{added_count} وزن پیش‌فرض اضافه شد.")

                elif isinstance(data, list):
                    # اگر فایل شامل لیستی از وزن‌ها باشد، آن‌ها را استفاده می‌کنیم
                    weight_sets = []
                    for idx, item in enumerate(data):
                        if all(key in item for key in ['w1', 'w2', 'w3']):
                            weight_name = item.get('name', f"وزن سفارشی {idx + 1}")
                            new_weight = {
                                'w1': item['w1'],
                                'w2': item['w2'],
                                'w3': item['w3'],
                                'name': weight_name
                            }

                            # بررسی تکراری نبودن قبل از اضافه کردن
                            if not is_duplicate_weight(new_weight, weight_sets):
                                weight_sets.append(new_weight)
                                print(f"وزن سفارشی {idx + 1} اضافه شد: {weight_name}")
                            else:
                                print(f"وزن سفارشی {idx + 1} تکراری بود و حذف شد: {weight_name}")

                    # اضافه کردن وزن‌های پیش‌فرض که تکراری نیستند
                    added_count = 0
                    for ws in default_weight_sets:
                        if not is_duplicate_weight(ws, weight_sets):
                            weight_sets.append(ws)
                            added_count += 1
                        else:
                            print(f"وزن پیش‌فرض تکراری حذف شد: {ws['name']}")

                    print(f"{added_count} وزن پیش‌فرض اضافه شد.")

                    if not weight_sets:
                        print("هیچ مجموعه وزن معتبری در فایل یافت نشد. از مقادیر پیش‌فرض استفاده می‌شود.")
                        weight_sets = default_weight_sets
                else:
                    print("ساختار فایل وزن‌ها معتبر نیست. از مقادیر پیش‌فرض استفاده می‌شود.")
                    weight_sets = default_weight_sets

                print(f"مجموعاً {len(weight_sets)} وزن برای اجرا آماده شد.")

            return weight_sets

        except json.JSONDecodeError:
            print("خطا در خواندن فایل JSON. از مقادیر پیش‌فرض استفاده می‌شود.")
        except Exception as e:
            print(f"خطا در بارگذاری وزن‌ها: {e}")
            print("از مقادیر پیش‌فرض استفاده می‌شود.")

    # اگر فایلی ارائه نشده یا در بارگذاری خطایی رخ داده، از مقادیر پیش‌فرض استفاده می‌کنیم
    print("استفاده از مجموعه وزن‌های پیش‌فرض...")
    return default_weight_sets