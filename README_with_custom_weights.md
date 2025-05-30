# راهنمای جامع استفاده از الگوریتم بهینه‌سازی تخصیص و زمان‌بندی واکسن

این راهنما نحوه استفاده از الگوریتم بهینه‌سازی تخصیص و زمان‌بندی واکسن را به همراه توضیحات دقیق درباره تأثیر تغییر پارامترهای مختلف بر نتایج شرح می‌دهد.

## فهرست مطالب
1. [مقدمه و ساختار الگوریتم](#1-مقدمه-و-ساختار-الگوریتم)
2. [نحوه استفاده](#2-نحوه-استفاده)
3. [پارامترهای کلیدی و تأثیر آنها](#3-پارامترهای-کلیدی-و-تأثیر-آنها)
4. [فایل‌های پیکربندی](#4-فایل‌های-پیکربندی)
5. [تفسیر نتایج](#5-تفسیر-نتایج)
6. [موارد خاص و رفع اشکال](#6-موارد-خاص-و-رفع-اشکال)

## 1. مقدمه و ساختار الگوریتم

الگوریتم ما یک مدل بهینه‌سازی برای تخصیص واکسن بین گروه‌های مختلف جمعیتی در شرایط اپیدمی است. هدف اصلی کمینه کردن مجموع هزینه‌های تأمین واکسن، هزینه‌های اجتماعی و هزینه‌های اقتصادی است.

### ساختار کلی:
- **ورودی**: داده‌های اپیدمی در فایل Excel و تنظیمات زمان‌بندی در فایل JSON
- **خروجی**: درصد بهینه واکسیناسیون هر گروه، زمان‌های بهینه و نمودارهای مختلف

### فرایند بهینه‌سازی:
1. استخراج داده‌های اپیدمی از فایل Excel
2. بررسی زمان‌های مختلف برای شروع واکسیناسیون (در صورت فعال بودن جستجوی زمان بهینه)
3. ساخت و حل مدل بهینه‌سازی برای هر ترکیب زمانی
4. انتخاب بهترین زمان‌بندی و درصدهای واکسیناسیون
5. تولید خروجی‌های گرافیکی و متنی

## 2. نحوه استفاده

### پیش‌نیازها:
```
Python 3.7+
pandas
numpy
matplotlib
pulp
tqdm
arabic_reshaper (اختیاری - برای نمایش متن فارسی)
python-bidi (اختیاری - برای نمایش متن فارسی)
```

### مراحل اجرا:

1. **آماده‌سازی فایل داده‌های اپیدمی**:
   - فایل Excel با نام `Group.xlsx` را در کنار اسکریپت قرار دهید
   - فایل باید ستون‌های `Time`, `S1`, `I1`, `Q1`, `V11`, `V21`, `R1`, `S2`, `I2`, `Q2`, `V12`, `V22`, `R2` را داشته باشد

2. **آماده‌سازی فایل پیکربندی (اختیاری)**:
   - فایل `config.json` را با تنظیمات زمان‌بندی ایجاد کنید
   - نمونه فایل پیکربندی:
   ```json
   {
     "tau1_group1_min": 5,
     "tau1_group1_max": 20,
     "tau1_group2_min": 5,
     "tau1_group2_max": 20,
     "gap_group1_min": 14,
     "gap_group1_max": 28,
     "gap_group2_min": 14,
     "gap_group2_max": 28,
     "time_step": 2
   }
   ```

3. **اجرای الگوریتم**:
   ```
   python VaccineAllocationOptimizer.py
   ```

4. **پاسخ به سوالات در زمان اجرا**:
   - آیا می‌خواهید از فایل پیکربندی استفاده کنید؟ (بله/خیر)
   - مسیر دقیق فایل `config.json` را وارد کنید (در صورت انتخاب بله در مرحله قبل)
   - آیا می‌خواهید جستجوی زمان‌های بهینه انجام شود؟ (بله/خیر) (در صورت استفاده از فایل پیکربندی)

## 3. پارامترهای کلیدی و تأثیر آنها

این بخش شرح می‌دهد هر پارامتر چه تأثیری بر نتایج دارد. با تغییر این پارامترها می‌توانید سناریوهای مختلف را بررسی کنید.

### پارامترهای هزینه (در بخش `prepare_data`):

| پارامتر | توضیح | مقدار فعلی | تأثیر افزایش | تأثیر کاهش |
|---------|-------|------------|--------------|------------|
| `self.P` | هزینه تأمین هر دوز واکسن از تولیدکننده | [10, 12] | افزایش هزینه کل، کاهش میزان استفاده از تولیدکننده گران‌تر | کاهش هزینه کل، افزایش احتمال استفاده از هر دو تولیدکننده |
| `self.SC` | هزینه اجتماعی هر گروه | [170, 130] | افزایش درصد واکسیناسیون گروه مربوطه | کاهش درصد واکسیناسیون گروه مربوطه |
| `self.Cq` | هزینه تعطیلی کسب و کار هر گروه | [0, 150] | تمایل به واکسیناسیون زودتر و بیشتر گروه مربوطه | تمایل به واکسیناسیون دیرتر و کمتر گروه مربوطه |
| `self.CV1` | هزینه ثابت واکسیناسیون دوز اول | 70 | کاهش درصد واکسیناسیون دوز اول | افزایش درصد واکسیناسیون دوز اول |
| `self.CV2` | هزینه ثابت واکسیناسیون دوز دوم | 50 | کاهش درصد واکسیناسیون دوز دوم | افزایش درصد واکسیناسیون دوز دوم |
| `self.L` | محدودیت تولید واکسن | 800 | افزایش کل واکسن تولیدی، افزایش درصد واکسیناسیون | کاهش کل واکسن تولیدی، کاهش درصد واکسیناسیون، افزایش تفاوت بین گروه‌ها |

### وزن‌های توابع هدف (در بخش `build_model`):

| پارامتر | توضیح | مقدار فعلی | تأثیر افزایش | تأثیر کاهش |
|---------|-------|------------|--------------|------------|
| `w1` | وزن هزینه تأمین واکسن | 0.35 | کاهش میزان استفاده از واکسن، تمایل به استفاده از تولیدکننده ارزان‌تر | افزایش میزان استفاده از واکسن، کمتر شدن اهمیت هزینه تولید |
| `w2` | وزن هزینه‌های اجتماعی | 0.35 | افزایش درصد واکسیناسیون گروه‌های با هزینه اجتماعی بالاتر | کاهش تمرکز بر هزینه‌های اجتماعی |
| `w3` | وزن هزینه‌های اقتصادی | 0.3 | افزایش اولویت‌بندی گروه 2 (کسب و کار)، شروع زودتر واکسیناسیون | کاهش اهمیت هزینه‌های اقتصادی در تصمیم‌گیری |

### محدودیت‌های حداقل واکسیناسیون (در بخش `build_model`):

| پارامتر | توضیح | مقدار فعلی | تأثیر افزایش | تأثیر کاهش |
|---------|-------|------------|--------------|------------|
| `U1[1] >= 0.15` | حداقل درصد دوز اول گروه 1 | 15% | اجبار به واکسیناسیون بیشتر گروه 1، افزایش هزینه | آزادی عمل بیشتر الگوریتم |
| `U1[2] >= 0.10` | حداقل درصد دوز اول گروه 2 | 10% | اجبار به واکسیناسیون بیشتر گروه 2، افزایش هزینه | آزادی عمل بیشتر الگوریتم |
| `U2[1] >= 0.10` | حداقل درصد دوز دوم گروه 1 | 10% | اجبار به تکمیل واکسیناسیون بیشتر گروه 1 | آزادی عمل بیشتر الگوریتم |
| `U2[2] >= 0.05` | حداقل درصد دوز دوم گروه 2 | 5% | اجبار به تکمیل واکسیناسیون بیشتر گروه 2 | آزادی عمل بیشتر الگوریتم |

### مثال‌های تغییر پارامترها و تأثیر آنها:

#### مثال 1: افزایش اهمیت گروه سالمندان
```python
# افزایش هزینه اجتماعی گروه 1
self.SC = [250, 130]  # افزایش از [170, 130]

# افزایش وزن هزینه‌های اجتماعی
w1, w2, w3 = 0.3, 0.5, 0.2  # افزایش w2 از 0.35 به 0.5
```
تأثیر: افزایش درصد واکسیناسیون گروه 1 (سالمندان)، احتمالاً شروع زودتر واکسیناسیون این گروه

#### مثال 2: افزایش اهمیت اقتصادی
```python
# افزایش هزینه تعطیلی کسب و کار
self.Cq = [0, 300]  # افزایش از [0, 150]

# افزایش وزن هزینه‌های اقتصادی
w1, w2, w3 = 0.3, 0.3, 0.4  # افزایش w3 از 0.3 به 0.4
```
تأثیر: افزایش درصد واکسیناسیون گروه 2 (کسب و کار)، احتمالاً شروع زودتر واکسیناسیون این گروه

#### مثال 3: کمبود شدید واکسن
```python
# کاهش محدودیت تولید واکسن
self.L = 400  # کاهش از 800 به 400
```
تأثیر: کاهش کلی درصدهای واکسیناسیون، تخصیص برابر با حداقل مقادیر مورد نیاز، افزایش تفاوت بین گروه‌ها


## استفاده از وزن‌های سفارشی برای توابع هدف

الگوریتم از سه تابع هدف برای بهینه‌سازی استفاده می‌کند:

- `Z1`: هزینه تأمین واکسن (تأثیر هزینه تولید)
- `Z2`: هزینه‌های اجتماعی (وابسته به درصد واکسیناسیون سالمندان)
- `Z3`: هزینه‌های اقتصادی (وابسته به درصد واکسیناسیون افراد دارای کسب‌وکار)

### نحوه استفاده از وزن‌های سفارشی

برای تنظیم وزن دلخواه برای هر تابع هدف، فایلی با فرمت `weights.json` در کنار اسکریپت قرار دهید. ساختار این فایل به صورت زیر است:

```json
{
  "name": "custom_weights_1",
  "w1": 0.2,
  "w2": 0.5,
  "w3": 0.3
}
```

- مجموع وزن‌ها باید برابر با 1 باشد.
- `w1` مربوط به وزن هزینه تأمین
- `w2` مربوط به وزن هزینه اجتماعی
- `w3` مربوط به وزن هزینه اقتصادی

در زمان اجرا، از شما پرسیده می‌شود که آیا مایل به استفاده از وزن سفارشی هستید یا خیر. در صورت تایید، مسیر فایل `weights.json` را وارد نمایید.


## 4. فایل‌های پیکربندی

### فایل پیکربندی زمان‌بندی (`config.json`):

| پارامتر | توضیح | مثال | تأثیر |
|---------|-------|------|-------|
| `tau1_group1_min` | حداقل زمان شروع دوز اول گروه 1 | 5 | تعیین زودترین زمان شروع واکسیناسیون گروه 1 |
| `tau1_group1_max` | حداکثر زمان شروع دوز اول گروه 1 | 20 | تعیین دیرترین زمان شروع واکسیناسیون گروه 1 |
| `tau1_group2_min` | حداقل زمان شروع دوز اول گروه 2 | 5 | تعیین زودترین زمان شروع واکسیناسیون گروه 2 |
| `tau1_group2_max` | حداکثر زمان شروع دوز اول گروه 2 | 20 | تعیین دیرترین زمان شروع واکسیناسیون گروه 2 |
| `gap_group1_min` | حداقل فاصله بین دوزها برای گروه 1 | 14 | تعیین کمترین فاصله زمانی بین دوز اول و دوم گروه 1 |
| `gap_group1_max` | حداکثر فاصله بین دوزها برای گروه 1 | 28 | تعیین بیشترین فاصله زمانی بین دوز اول و دوم گروه 1 |
| `gap_group2_min` | حداقل فاصله بین دوزها برای گروه 2 | 14 | تعیین کمترین فاصله زمانی بین دوز اول و دوم گروه 2 |
| `gap_group2_max` | حداکثر فاصله بین دوزها برای گروه 2 | 28 | تعیین بیشترین فاصله زمانی بین دوز اول و دوم گروه 2 |
| `time_step` | گام زمانی جستجو | 2 | تعیین دقت جستجوی زمان - مقادیر کوچکتر دقت بیشتر اما زمان اجرای طولانی‌تر |

### تنظیمات مختلف فایل پیکربندی:

#### زمان‌بندی بسیار زودهنگام
```json
{
  "tau1_group1_min": 1,
  "tau1_group1_max": 10,
  "tau1_group2_min": 1,
  "tau1_group2_max": 10,
  "gap_group1_min": 14,
  "gap_group1_max": 21,
  "gap_group2_min": 14,
  "gap_group2_max": 21,
  "time_step": 2
}
```
تأثیر: بررسی تأثیر شروع بسیار زودهنگام واکسیناسیون

#### اولویت‌بندی گروه کسب و کار
```json
{
  "tau1_group1_min": 15,
  "tau1_group1_max": 25,
  "tau1_group2_min": 5,
  "tau1_group2_max": 15,
  "gap_group1_min": 14,
  "gap_group1_max": 21,
  "gap_group2_min": 14,
  "gap_group2_max": 21,
  "time_step": 2
}
```
تأثیر: بررسی تأثیر شروع زودتر واکسیناسیون گروه 2 نسبت به گروه 1

#### فاصله‌های متفاوت بین دوزها
```json
{
  "tau1_group1_min": 10,
  "tau1_group1_max": 20,
  "tau1_group2_min": 10,
  "tau1_group2_max": 20,
  "gap_group1_min": 21,
  "gap_group1_max": 28,
  "gap_group2_min": 14,
  "gap_group2_max": 18,
  "time_step": 2
}
```
تأثیر: بررسی تأثیر فاصله طولانی‌تر برای گروه 1 و فاصله کوتاه‌تر برای گروه 2

## 5. تفسیر نتایج

نتایج الگوریتم شامل چندین بخش مهم است:

### زمان‌های بهینه:
- زمان شروع دوز اول برای هر گروه: نشان‌دهنده روز بهینه برای شروع واکسیناسیون
- زمان شروع دوز دوم برای هر گروه: نشان‌دهنده روز بهینه برای شروع دوز دوم
- فاصله بین دوزها: نشان‌دهنده استراتژی فاصله‌گذاری

### درصد واکسیناسیون:
- درصد دوز اول هر گروه: نسبت افراد مستعد که دوز اول را دریافت می‌کنند
- درصد دوز دوم هر گروه: نسبت افراد واکسینه شده با دوز اول که دوز دوم را دریافت می‌کنند

### تخصیص تولید:
- تعداد واکسن تولید شده توسط هر تولیدکننده: نحوه توزیع تولید بین تولیدکنندگان

### هزینه کل:
- مقدار تابع هدف: نشان‌دهنده کل هزینه‌های ترکیبی (تأمین، اجتماعی و اقتصادی)

### محاسبات تکمیلی:
- تعداد واکسن مورد نیاز برای هر گروه: تعداد واقعی دوزهای تخصیص داده شده
- درصد استفاده از ظرفیت تولید: نسبت استفاده از حداکثر ظرفیت تولید

### الگوهای نتایج و تفسیر آنها:

#### الگوی 1: درصدهای مساوی دوز اول و دوم برای گروه 2
اگر درصد دوز اول و دوم برای گروه 2 یکسان است:
- نشان‌دهنده اهمیت ایمنی کامل برای افراد دارای کسب و کار
- هزینه‌های اقتصادی تعطیلی تأثیر بالایی دارند
- واکسیناسیون کامل (هر دو دوز) برای این گروه صرفه اقتصادی دارد

#### الگوی 2: درصد بالاتر دوز اول نسبت به دوز دوم برای گروه 1
اگر درصد دوز اول گروه 1 بیشتر از درصد دوز دوم است:
- نشان‌دهنده استراتژی "اول پهنا، بعد عمق" برای گروه‌های پرخطر
- ترجیح به پوشش وسیع‌تر افراد با حداقل یک دوز به جای پوشش کامل تعداد کمتری
- محدودیت منابع نقش مهمی در این تصمیم دارد

#### الگوی 3: استفاده کامل از ظرفیت تولید
اگر 100% ظرفیت تولید استفاده می‌شود:
- نشان‌دهنده کمبود منابع واکسن
- هزینه‌های اجتماعی و اقتصادی بیماری بیشتر از هزینه تولید واکسن است
- الگوریتم تمایل به استفاده حداکثری از منابع دارد

## 6. موارد خاص و رفع اشکال

### مشکل: نتایج همواره 100% واکسیناسیون
**علت**: زمان شروع واکسیناسیون دیر است (معمولاً بالای 30) و تا آن زمان تعداد افراد مستعد باقی‌مانده بسیار کم شده است.

**راه حل**: 
- استفاده از زمان‌های شروع زودتر (10-15)
- کاهش محدودیت تولید واکسن (L)
- افزایش هزینه‌های ثابت واکسیناسیون (CV1, CV2)

### مشکل: خطای "Infeasible" (غیرقابل حل)
**علت**: تناقض در محدودیت‌ها، معمولاً ناشی از زمان‌های نامناسب یا محدودیت‌های حداقل واکسیناسیون بالا

**راه حل**:
- بررسی فاصله زمانی بین دوز اول و دوم
- کاهش محدودیت‌های حداقل واکسیناسیون
- افزایش محدودیت تولید واکسن

### مشکل: زمان اجرای بسیار طولانی
**علت**: تعداد زیاد ترکیب‌های زمانی برای بررسی

**راه حل**:
- افزایش گام زمانی (`time_step`) در فایل پیکربندی
- کوچک کردن محدوده‌های زمانی جستجو
- محدود کردن فاصله‌های بین دوزها

### مشکل: نتایج غیرمنتظره یا غیرمنطقی
**علت**: معمولاً ناشی از تنظیم نامناسب پارامترها یا تناقض در داده‌های ورودی

**راه حل**:
- بررسی فایل Excel برای اطمینان از صحت داده‌ها
- تنظیم مجدد وزن‌های توابع هدف به مقادیر متعادل (مثلاً 0.33, 0.33, 0.34)
- تنظیم تناسب بین هزینه‌های مختلف (P، SC، Cq، CV1، CV2)