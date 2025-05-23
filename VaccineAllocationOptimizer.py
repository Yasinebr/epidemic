#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import *
import json
import itertools
from tqdm import tqdm
from weight_sum_module import load_weights, load_multiple_weights


class VaccineAllocationOptimizer:
    """
    کلاس بهینه‌سازی تخصیص واکسن در شرایط اپیدمی با قابلیت بهینه‌سازی زمان‌بندی
    """

    def __init__(self, excel_file_path, config_file_path=None, weights_file_path=None):
        """
        مقداردهی اولیه با داده‌های فایل اکسل و فایل پیکربندی

        پارامترها:
            excel_file_path (str): مسیر فایل اکسل حاوی داده‌های اپیدمی
            config_file_path (str): مسیر فایل پیکربندی JSON (اختیاری)
            weights_file_path (str): مسیر فایل وزن‌های توابع هدف (اختیاری)
        """
        print(f"در حال بارگذاری داده‌های اپیدمی از فایل: {excel_file_path}")

        # بررسی وجود فایل اکسل
        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"فایل {excel_file_path} یافت نشد.")

        # خواندن داده‌های اپیدمی
        try:
            self.data = pd.read_excel(excel_file_path)
            print(f"داده‌ها با موفقیت بارگذاری شدند. تعداد نقاط زمانی: {len(self.data)}")

            # بررسی وجود ستون‌های مورد نیاز
            required_columns = ['Time', 'S1', 'I1', 'Q1', 'V11', 'V21', 'R1',
                                'S2', 'I2', 'Q2', 'V12', 'V22', 'R2']

            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"ستون‌های مورد نیاز در فایل وجود ندارند: {missing_columns}")

            # نمایش داده‌های اولیه
            print("\n=== نمونه‌ای از داده‌های اپیدمی (5 نقطه زمانی اول) ===")
            print(self.data.head())

            # بارگذاری وزن‌های توابع هدف
            self.w1, self.w2, self.w3, self.weights_valid, self.weight_name = load_weights(weights_file_path)

            # آماده‌سازی داده‌ها برای استفاده در مدل
            self.prepare_data()

            # بارگیری تنظیمات از فایل config.json (اگر ارائه شده باشد)
            self.config = None
            if config_file_path:
                self.load_config(config_file_path)
                print("تنظیمات زمان‌بندی از فایل پیکربندی بارگذاری شد.")
            else:
                print("فایل پیکربندی ارائه نشده است. از زمان‌بندی پیش‌فرض استفاده می‌شود.")

        except Exception as e:
            print(f"خطا در خواندن فایل اکسل: {e}")
            raise

    def load_config(self, config_file_path):
        """
        بارگیری تنظیمات از فایل JSON

        پارامترها:
            config_file_path (str): مسیر فایل پیکربندی JSON
        """
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"فایل پیکربندی {config_file_path} یافت نشد.")

        try:
            with open(config_file_path, 'r', encoding='utf-8') as file:
                self.config = json.load(file)

            # بررسی وجود کلیدهای مورد نیاز
            required_keys = [
                "tau1_group1_min", "tau1_group1_max",
                "tau1_group2_min", "tau1_group2_max",
                "gap_group1_min", "gap_group1_max",
                "gap_group2_min", "gap_group2_max",
                "time_step"
            ]

            missing_keys = [key for key in required_keys if key not in self.config]
            if missing_keys:
                raise ValueError(f"کلیدهای زیر در فایل پیکربندی وجود ندارند: {missing_keys}")

            print("\n=== تنظیمات زمان‌بندی ===")
            print(
                f"محدوده شروع دوز اول برای گروه 1: {self.config['tau1_group1_min']} تا {self.config['tau1_group1_max']}")
            print(
                f"محدوده شروع دوز اول برای گروه 2: {self.config['tau1_group2_min']} تا {self.config['tau1_group2_max']}")
            print(
                f"محدوده فاصله بین دوزها برای گروه 1: {self.config['gap_group1_min']} تا {self.config['gap_group1_max']}")
            print(
                f"محدوده فاصله بین دوزها برای گروه 2: {self.config['gap_group2_min']} تا {self.config['gap_group2_max']}")
            print(f"گام زمانی: {self.config['time_step']}")

        except json.JSONDecodeError as e:
            raise ValueError(f"خطا در خواندن فایل JSON: {e}")
        except Exception as e:
            raise Exception(f"خطا در بارگیری تنظیمات: {e}")

    def prepare_data(self):
        """
        آماده‌سازی داده‌ها برای استفاده در مدل
        """
        print("\nدر حال آماده‌سازی داده‌ها برای استفاده در مدل...")

        # استخراج زمان‌ها
        self.time_points = self.data['Time'].values

        # تعداد نقاط زمانی
        self.T = len(self.time_points)

        # تعداد گروه‌ها و تولیدکنندگان
        self.num_groups = 2
        self.num_manufacturers = 2

        # داده‌های مربوط به گروه‌ها
        self.S = [self.data['S1'].values, self.data['S2'].values]  # افراد مستعد
        self.I = [self.data['I1'].values, self.data['I2'].values]  # افراد آلوده
        self.Q = [self.data['Q1'].values, self.data['Q2'].values]  # افراد قرنطینه شده
        self.V1 = [self.data['V11'].values, self.data['V12'].values]  # افراد واکسینه شده (دوز اول)
        self.V2 = [self.data['V21'].values, self.data['V22'].values]  # افراد واکسینه شده (دوز دوم)
        self.R = [self.data['R1'].values, self.data['R2'].values]  # افراد بهبود یافته

        # زمان شروع واکسیناسیون دوز اول برای هر گروه (مقادیر تصحیح شده)
        self.tau1 = [30, 35]  # تصحیح شده: از روز 30 شروع می‌شود

        # زمان شروع واکسیناسیون دوز دوم برای هر گروه (مقادیر تصحیح شده)
        self.tau2 = [75, 80]  # تصحیح شده: فاصله 45 روز بین دوزها

        # زمان اتمام اپیدمی
        self.end_time = [self.T - 1, self.T - 1]  # فرض می‌کنیم آخرین نقطه زمانی، پایان اپیدمی است

        # پارامترهای هزینه - متعادل شده برای تأثیر بهتر وزن‌ها
        self.P = [8, 6]  # هزینه تأمین هر دوز واکسن - تولیدکننده دوم ارزان‌تر است
        self.SC = [300, 300]  # هزینه‌های اجتماعی متعادل شده (به جای [450, 220])
        self.Cq = [200, 220]  # هزینه قرنطینه متعادل‌تر
        self.CV1 = 50  # هزینه ثابت واکسیناسیون دوز اول
        self.CV2 = 30  # هزینه ثابت واکسیناسیون دوز دوم
        self.L = 3000  # تصحیح شده: افزایش ظرفیت تولید برای نتایج واقعی‌تر

        # پارامترهای تابع هزینه قرنطینه
        self.A = 15
        self.B = 40

        # بررسی مقادیر
        print("\n=== اطلاعات آماده‌سازی شده ===")
        print(f"تعداد نقاط زمانی: {self.T}")
        print(f"تعداد گروه‌ها: {self.num_groups}")
        print(f"تعداد تولیدکنندگان: {self.num_manufacturers}")
        print(f"زمان شروع واکسیناسیون دوز اول (تصحیح شده): {self.tau1}")
        print(f"زمان شروع واکسیناسیون دوز دوم (تصحیح شده): {self.tau2}")
        print(f"زمان اتمام اپیدمی: {self.end_time}")
        print(f"ظرفیت تولید تصحیح شده: {self.L}")
        print(f"هزینه‌های اجتماعی متعادل شده: {self.SC}")

        # بررسی جمعیت‌ها
        total_pop_group1 = self.S[0][0] + self.I[0][0] + self.Q[0][0] + self.V1[0][0] + self.V2[0][0] + self.R[0][0]
        total_pop_group2 = self.S[1][0] + self.I[1][0] + self.Q[1][0] + self.V1[1][0] + self.V2[1][0] + self.R[1][0]

        print(f"جمعیت کل گروه 1 (نقطه زمانی اول): {total_pop_group1}")
        print(f"جمعیت کل گروه 2 (نقطه زمانی اول): {total_pop_group2}")

    def build_model(self, tau1=None, tau2=None):
        if tau1 is not None:
            self.tau1 = tau1
        if tau2 is not None:
            self.tau2 = tau2

        print("\nدر حال ساخت مدل بهینه‌سازی...")
        print(f"زمان‌های استفاده شده برای دوز اول: {self.tau1}")
        print(f"زمان‌های استفاده شده برای دوز دوم: {self.tau2}")
        print("🎯 محدودیت‌های انعطاف‌پذیر اعمال می‌شود...")

        if hasattr(self, 'model'):
            del self.model

        self.model = LpProblem("Vaccine_Allocation_Optimization", LpMinimize)

        self.U1 = {j: LpVariable(f"U1_{j}", 0, 1) for j in range(1, self.num_groups + 1)}
        self.U2 = {j: LpVariable(f"U2_{j}", 0, 1) for j in range(1, self.num_groups + 1)}
        self.V_prime = {i: LpVariable(f"V_prime_{i}", lowBound=0) for i in range(1, self.num_manufacturers + 1)}

        self.objective1 = lpSum(self.P[i - 1] * self.V_prime[i] for i in range(1, self.num_manufacturers + 1))

        self.objective2 = 0
        for j in range(1, self.num_groups + 1):
            j_idx = j - 1
            social_cost_before_vax = self.SC[j_idx] * sum(self.I[j_idx][t] for t in range(self.tau1[j_idx]))
            total_infected_between_doses = sum(self.I[j_idx][t] for t in range(self.tau1[j_idx], self.tau2[j_idx]))
            social_cost_between_doses = (
                    self.SC[j_idx] * total_infected_between_doses * (1 - 0.7 * self.U1[j]) +
                    self.CV1 * 1.5 * self.U1[j]
            )
            total_infected_after_dose2 = sum(
                self.I[j_idx][t] for t in range(self.tau2[j_idx], self.end_time[j_idx] + 1))
            social_cost_after_dose2 = (
                    self.SC[j_idx] * total_infected_after_dose2 * (1 - 0.9 * self.U2[j]) +
                    self.CV2 * 1.5 * self.U2[j]
            )
            self.objective2 += social_cost_before_vax + social_cost_between_doses + social_cost_after_dose2

        self.objective3 = 0
        for j in range(1, self.num_groups + 1):
            j_idx = j - 1
            # وزن‌های اقتصادی متعادل‌تر (تغییر اصلی)
            economic_weight = 0.8 if j == 2 else 0.7  # به جای 1.0 vs 0.4
            total_people_before_vax = sum(
                self.S[j_idx][t] + self.I[j_idx][t] + self.Q[j_idx][t]
                for t in range(self.tau1[j_idx])
            )
            total_infected_between_doses = sum(
                self.I[j_idx][t]
                for t in range(self.tau1[j_idx], self.tau2[j_idx])
            )
            total_infected_after_dose2 = sum(
                self.I[j_idx][t]
                for t in range(self.tau2[j_idx], self.end_time[j_idx] + 1)
            )
            Cq_before_vax = self.A * self.tau1[j_idx] + self.B
            economic_cost_before_vax = Cq_before_vax * total_people_before_vax * economic_weight
            Cq_between_doses = self.A * (self.tau2[j_idx] - self.tau1[j_idx]) + self.B
            economic_cost_between_doses = Cq_between_doses * total_infected_between_doses * (
                    1 - 0.7 * self.U1[j]) * economic_weight
            Cq_after_dose2 = self.A * (self.end_time[j_idx] - self.tau2[j_idx]) + self.B
            economic_cost_after_dose2 = Cq_after_dose2 * total_infected_after_dose2 * (
                    1 - 0.9 * self.U2[j]) * economic_weight
            self.objective3 += economic_cost_before_vax + economic_cost_between_doses + economic_cost_after_dose2

        norm_factor1 = 5000
        norm_factor2 = 400000
        norm_factor3 = 10000000

        normalized_objective1 = self.objective1 / norm_factor1
        normalized_objective2 = self.objective2 / norm_factor2
        normalized_objective3 = self.objective3 / norm_factor3

        combined_objective = self.w1 * normalized_objective1 + self.w2 * normalized_objective2 + self.w3 * normalized_objective3

        self.model += combined_objective

        self.original_objective1 = self.objective1
        self.original_objective2 = self.objective2
        self.original_objective3 = self.objective3

        total_vax_group1 = 0
        total_vax_group2 = 0
        for j in range(1, self.num_groups + 1):
            j_idx = j - 1
            total_susceptible = sum(self.S[j_idx][t] for t in range(self.tau1[j_idx], self.tau2[j_idx]))
            total_vaccinated_dose1 = sum(self.V1[j_idx][t] for t in range(self.tau2[j_idx], self.end_time[j_idx] + 1))
            group_vax_need = (
                    self.U1[j] * total_susceptible +
                    self.U2[j] * total_vaccinated_dose1
            )
            if j == 1:
                total_vax_group1 += group_vax_need
            else:
                total_vax_group2 += group_vax_need

        total_vax_all = total_vax_group1 + total_vax_group2

        # محدودیت‌های تخصیص کلی خیلی نرم‌تر (تغییر اصلی)
        self.model += total_vax_group1 >= 0.20 * total_vax_all, "Min_Vax_Allocation_Group1"  # کاهش از 0.4
        self.model += total_vax_group2 >= 0.20 * total_vax_all, "Min_Vax_Allocation_Group2"  # کاهش از 0.6

        self.model += total_vax_all <= lpSum(
            self.V_prime[i] for i in range(1, self.num_manufacturers + 1)), "Vaccine_Supply_Demand_Balance"
        self.model += lpSum(
            self.V_prime[i] for i in range(1, self.num_manufacturers + 1)) <= self.L, "Production_Capacity"

        # محدودیت‌های حداقل خیلی نرم (تغییر اصلی)
        self.model += self.U1[1] >= 0.05, "Min_Vaccination_Group1_Dose1"  # کاهش از 0.15
        self.model += self.U1[2] >= 0.05, "Min_Vaccination_Group2_Dose1"  # کاهش از 0.20
        self.model += self.U2[1] >= 0.05, "Min_Vaccination_Group1_Dose2"  # کاهش از 0.10
        self.model += self.U2[2] >= 0.05, "Min_Vaccination_Group2_Dose2"  # کاهش از 0.15

        # محدودیت‌های حداکثر خیلی نرم (تغییر اصلی)
        self.model += self.U1[1] <= 0.95, "Max_Vaccination_Group1_Dose1"  # افزایش از 0.70
        self.model += self.U2[1] <= 0.95, "Max_Vaccination_Group1_Dose2"  # افزایش از 0.65
        self.model += self.U1[2] <= 0.95, "Max_Vaccination_Group2_Dose1"  # افزایش از 0.70
        self.model += self.U2[2] <= 0.95, "Max_Vaccination_Group2_Dose2"  # افزایش از 0.65

        self.model += self.U2[1] <= self.U1[1], "Dose2_Limit_Group1"
        self.model += self.U2[2] <= self.U1[2], "Dose2_Limit_Group2"

        total_production = lpSum(self.V_prime[i] for i in range(1, self.num_manufacturers + 1))
        # محدودیت‌های تولیدکنندگان نرم‌تر (تغییر اصلی)
        self.model += self.V_prime[1] >= 0.10 * total_production, "Min_Producer1"  # کاهش از 0.25
        self.model += self.V_prime[1] <= 0.90 * total_production, "Max_Producer1"  # افزایش از 0.75
        self.model += self.V_prime[2] >= 0.10 * total_production, "Min_Producer2"  # کاهش از 0.25
        self.model += self.V_prime[2] <= 0.90 * total_production, "Max_Producer2"  # افزایش از 0.75

        # محدودیت‌های نسبت خیلی نرم (تغییر اصلی)
        self.model += self.U1[2] <= 10.0 * self.U1[1], "Max_Ratio_Group2_Dose1"  # افزایش از 3.0
        self.model += self.U2[2] <= 10.0 * self.U2[1], "Max_Ratio_Group2_Dose2"  # افزایش از 3.0

        # محدودیت اختلاف کل خیلی نرم (تغییر اصلی)
        diff = LpVariable("Difference_U", lowBound=0)
        self.model += self.U1[2] + self.U2[2] - self.U1[1] - self.U2[1] <= diff
        self.model += self.U1[1] + self.U2[1] - self.U1[2] - self.U2[2] <= diff
        self.model += diff <= 0.9, "Max_Total_Vaccine_Diff"  # افزایش از 0.3

        print("مدل بهینه‌سازی با محدودیت‌های انعطاف‌پذیر ساخته شد.")
        print("حالا وزن‌ها تأثیر واقعی خود را خواهند داشت! ✅")

    def solve_model(self):
        """
        حل مدل بهینه‌سازی
        """
        print("\nدر حال حل مدل بهینه‌سازی...")

        # حل مدل
        self.model.solve(PULP_CBC_CMD(msg=False))

        # بررسی وضعیت حل
        status = LpStatus[self.model.status]
        print(f"\nوضعیت حل: {status}")

        if self.model.status == LpStatusOptimal:
            print("مدل با موفقیت حل شد.")

            # نمایش نتایج
            print("\n=== نتایج بهینه‌سازی ===")

            print("\n--- نسبت واکسن دوز اول تخصیص داده شده به هر گروه ---")
            for j in range(1, self.num_groups + 1):
                group_name = "افراد بالای 60 سال" if j == 1 else "افراد دارای کسب و کار"
                print(f"گروه {j} ({group_name}): {value(self.U1[j]):.4f}")

            print("\n--- نسبت واکسن دوز دوم تخصیص داده شده به هر گروه ---")
            for j in range(1, self.num_groups + 1):
                group_name = "افراد بالای 60 سال" if j == 1 else "افراد دارای کسب و کار"
                print(f"گروه {j} ({group_name}): {value(self.U2[j]):.4f}")

            print("\n--- تعداد واکسن تولید شده توسط هر تولیدکننده ---")
            for i in range(1, self.num_manufacturers + 1):
                print(f"تولیدکننده {i}: {value(self.V_prime[i]):.2f}")

            # مقادیر توابع هدف اصلی - استفاده از مقادیر غیرنرمال‌شده برای گزارش
            objective1_value = value(self.original_objective1)
            objective2_value = value(self.original_objective2)
            objective3_value = value(self.original_objective3)
            total_objective_value = value(self.model.objective)  # این نرمال‌شده است

            print("\n--- مقادیر توابع هدف ---")
            print(
                f"وزن‌های استفاده شده در این اجرا: w1={self.w1:.2f}, w2={self.w2:.2f}, w3={self.w3:.2f} ({self.weight_name})")
            print(f"Z1 (هزینه تأمین واکسن): {objective1_value:.2f} (وزن: {self.w1:.2f})")
            print(f"Z2 (هزینه‌های اجتماعی): {objective2_value:.2f} (وزن: {self.w2:.2f})")
            print(f"Z3 (هزینه‌های اقتصادی): {objective3_value:.2f} (وزن: {self.w3:.2f})")
            print(f"مقدار تابع هدف کل (نرمال‌شده): {total_objective_value:.2f}")

            # اضافه کردن تحلیل مقایسه‌ای تولیدکنندگان
            print("\n--- تحلیل مقایسه‌ای تولیدکنندگان ---")
            producer1_cost = value(self.V_prime[1]) * self.P[0]
            producer2_cost = value(self.V_prime[2]) * self.P[1]
            total_cost = producer1_cost + producer2_cost

            print(
                f"تولیدکننده 1: تعداد {value(self.V_prime[1]):.2f} واکسن با هزینه کل {producer1_cost:.2f} ({producer1_cost / total_cost * 100:.1f}% از کل)")
            print(
                f"تولیدکننده 2: تعداد {value(self.V_prime[2]):.2f} واکسن با هزینه کل {producer2_cost:.2f} ({producer2_cost / total_cost * 100:.1f}% از کل)")

            if self.P[0] < self.P[1]:
                print("تولیدکننده 1 ارزان‌تر است.")
            else:
                print("تولیدکننده 2 ارزان‌تر است.")

            if value(self.V_prime[1]) > value(self.V_prime[2]):
                print("بیشترین تولید از تولیدکننده 1 است.")
            else:
                print("بیشترین تولید از تولیدکننده 2 است.")

            # اضافه کردن آنالیز تأثیر وزن‌ها
            print("\n--- تحلیل تأثیر وزن‌ها بر نتایج ---")
            print(f"وزن هزینه تأمین (w1): {self.w1:.2f} -> تأثیر بر انتخاب تولیدکننده")
            print(f"وزن هزینه اجتماعی (w2): {self.w2:.2f} -> تأثیر بر میزان واکسیناسیون گروه 1")
            print(f"وزن هزینه اقتصادی (w3): {self.w3:.2f} -> تأثیر بر میزان واکسیناسیون گروه 2")

            # اضافه کردن تحلیل عدالت تخصیص
            print("\n--- تحلیل عدالت تخصیص واکسن ---")
            # محاسبه کل افراد مستعد در هر گروه
            total_susceptible_group1 = sum(self.S[0][t] for t in range(self.tau1[0], self.tau2[0]))
            total_susceptible_group2 = sum(self.S[1][t] for t in range(self.tau1[1], self.tau2[1]))

            # محاسبه کل افراد واکسینه شده دوز اول در هر گروه
            total_vaccinated_dose1_group1 = sum(self.V1[0][t] for t in range(self.tau2[0], self.end_time[0] + 1))
            total_vaccinated_dose1_group2 = sum(self.V1[1][t] for t in range(self.tau2[1], self.end_time[1] + 1))

            # محاسبه تعداد واکسن تخصیص یافته به هر گروه
            vaccine_dose1_group1 = value(self.U1[1]) * total_susceptible_group1
            vaccine_dose1_group2 = value(self.U1[2]) * total_susceptible_group2
            vaccine_dose2_group1 = value(self.U2[1]) * total_vaccinated_dose1_group1
            vaccine_dose2_group2 = value(self.U2[2]) * total_vaccinated_dose1_group2

            # محاسبه درصدها
            total_dose1 = vaccine_dose1_group1 + vaccine_dose1_group2
            total_dose2 = vaccine_dose2_group1 + vaccine_dose2_group2

            print(
                f"تعداد واکسن دوز اول برای گروه 1 (افراد بالای 60 سال): {vaccine_dose1_group1:.2f} ({vaccine_dose1_group1 / total_dose1 * 100:.1f}%)")
            print(
                f"تعداد واکسن دوز اول برای گروه 2 (افراد دارای کسب و کار): {vaccine_dose1_group2:.2f} ({vaccine_dose1_group2 / total_dose1 * 100:.1f}%)")
            print(
                f"تعداد واکسن دوز دوم برای گروه 1 (افراد بالای 60 سال): {vaccine_dose2_group1:.2f} ({vaccine_dose2_group1 / total_dose2 * 100:.1f}%)")
            print(
                f"تعداد واکسن دوز دوم برای گروه 2 (افراد دارای کسب و کار): {vaccine_dose2_group2:.2f} ({vaccine_dose2_group2 / total_dose2 * 100:.1f}%)")

            # بررسی شاخص عدالت - اختلاف بین نسبت‌های واکسیناسیون
            equity_diff_dose1 = abs(value(self.U1[1]) - value(self.U1[2]))
            equity_diff_dose2 = abs(value(self.U2[1]) - value(self.U2[2]))

            print(
                f"شاخص عدالت - اختلاف نسبت واکسیناسیون دوز اول: {equity_diff_dose1:.4f} ({equity_diff_dose1 * 100:.1f}%)")
            print(
                f"شاخص عدالت - اختلاف نسبت واکسیناسیون دوز دوم: {equity_diff_dose2:.4f} ({equity_diff_dose2 * 100:.1f}%)")

            # بررسی تناسب با جمعیت
            population_ratio_group1 = self.S[0][0] / (self.S[0][0] + self.S[1][0])
            allocation_ratio_dose1 = vaccine_dose1_group1 / total_dose1
            allocation_ratio_dose2 = vaccine_dose2_group1 / total_dose2

            print(f"نسبت جمعیت گروه 1: {population_ratio_group1:.4f} ({population_ratio_group1 * 100:.1f}%)")
            print(
                f"نسبت تخصیص واکسن دوز اول به گروه 1: {allocation_ratio_dose1:.4f} ({allocation_ratio_dose1 * 100:.1f}%)")
            print(
                f"نسبت تخصیص واکسن دوز دوم به گروه 1: {allocation_ratio_dose2:.4f} ({allocation_ratio_dose2 * 100:.1f}%)")

            population_effectiveness = min(allocation_ratio_dose1 / population_ratio_group1, 1.0)
            print(
                f"شاخص کارایی عدالت (نسبت به جمعیت گروه 1): {population_effectiveness:.4f} ({population_effectiveness * 100:.1f}%)")

            return {
                'U1': {j: value(self.U1[j]) for j in range(1, self.num_groups + 1)},
                'U2': {j: value(self.U2[j]) for j in range(1, self.num_groups + 1)},
                'V_prime': {i: value(self.V_prime[i]) for i in range(1, self.num_manufacturers + 1)},
                'objective_value': total_objective_value,
                'objective1_value': objective1_value,
                'objective2_value': objective2_value,
                'objective3_value': objective3_value,
                'weights': {'w1': self.w1, 'w2': self.w2, 'w3': self.w3},
                'tau1': self.tau1,
                'tau2': self.tau2,
                'equity_metrics': {
                    'vaccine_dose1_group1': vaccine_dose1_group1,
                    'vaccine_dose1_group2': vaccine_dose1_group2,
                    'vaccine_dose2_group1': vaccine_dose2_group1,
                    'vaccine_dose2_group2': vaccine_dose2_group2,
                    'equity_diff_dose1': equity_diff_dose1,
                    'equity_diff_dose2': equity_diff_dose2,
                    'population_effectiveness': population_effectiveness
                }
            }
        else:
            print("مدل به جواب بهینه نرسید.")
            print("دلیل عدم موفقیت:", LpStatus[self.model.status])

            return None

    def calculate_additional_info(self):
        """
        محاسبه اطلاعات تکمیلی و تفسیر نتایج
        """
        print("\n=== محاسبات تکمیلی ===")

        # محاسبه کل واکسن مورد نیاز
        total_vax_need = 0
        for j in range(1, self.num_groups + 1):
            j_idx = j - 1  # اندیس آرایه

            # مجموع افراد مستعد بین زمان شروع دوز اول و دوز دوم
            total_susceptible = sum(self.S[j_idx][t] for t in range(self.tau1[j_idx], self.tau2[j_idx]))

            # مجموع افراد واکسینه شده دوز اول بین زمان شروع دوز دوم و پایان اپیدمی
            total_vaccinated_dose1 = sum(self.V1[j_idx][t] for t in range(self.tau2[j_idx], self.end_time[j_idx] + 1))

            # واکسن مورد نیاز برای این گروه
            group_vax_need = (
                    value(self.U1[j]) * total_susceptible +  # نیاز به واکسن دوز اول
                    value(self.U2[j]) * total_vaccinated_dose1  # نیاز به واکسن دوز دوم
            )

            group_name = "افراد بالای 60 سال" if j == 1 else "افراد دارای کسب و کار"
            print(f"واکسن مورد نیاز برای گروه {j} ({group_name}):")
            print(f"  - دوز اول: {value(self.U1[j]) * total_susceptible:.2f}")
            print(f"  - دوز دوم: {value(self.U2[j]) * total_vaccinated_dose1:.2f}")
            print(f"  - مجموع: {group_vax_need:.2f}")

            total_vax_need += group_vax_need

        # کل واکسن تولید شده
        total_production = sum(value(self.V_prime[i]) for i in range(1, self.num_manufacturers + 1))

        print(f"\nکل واکسن مورد نیاز: {total_vax_need:.2f}")
        print(f"کل واکسن تولید شده: {total_production:.2f}")

        # بررسی استفاده از ظرفیت تولید
        capacity_usage = total_production / self.L * 100
        print(f"درصد استفاده از ظرفیت تولید: {capacity_usage:.2f}%")

        # تفسیر نتایج
        print("\n=== تفسیر نتایج ===")

        # تفسیر اولویت‌بندی گروه‌ها
        u1_group1 = value(self.U1[1])
        u1_group2 = value(self.U1[2])

        if u1_group1 > u1_group2:
            print("گروه 1 (افراد بالای 60 سال) در اولویت بالاتری برای دریافت واکسن دوز اول قرار دارند.")
        elif u1_group2 > u1_group1:
            print("گروه 2 (افراد دارای کسب و کار) در اولویت بالاتری برای دریافت واکسن دوز اول قرار دارند.")
        else:
            print("هر دو گروه اولویت یکسانی برای دریافت واکسن دوز اول دارند.")

        # تفسیر تولیدکنندگان
        v_prime_1 = value(self.V_prime[1])
        v_prime_2 = value(self.V_prime[2])

        if v_prime_1 > v_prime_2:
            print("تولیدکننده 1 سهم بیشتری در تولید واکسن دارد.")
        elif v_prime_2 > v_prime_1:
            print("تولیدکننده 2 سهم بیشتری در تولید واکسن دارد.")
        else:
            print("هر دو تولیدکننده سهم یکسانی در تولید واکسن دارند.")

        # تفسیر مقایسه هزینه تولیدکنندگان
        producer1_unit_cost = self.P[0]
        producer2_unit_cost = self.P[1]
        print(f"\nهزینه واحد تولیدکننده 1: {producer1_unit_cost}")
        print(f"هزینه واحد تولیدکننده 2: {producer2_unit_cost}")

        if producer1_unit_cost < producer2_unit_cost:
            print("از نظر قیمت، تولیدکننده 1 ارزان‌تر است.")
        elif producer2_unit_cost < producer1_unit_cost:
            print("از نظر قیمت، تولیدکننده 2 ارزان‌تر است.")
        else:
            print("هر دو تولیدکننده قیمت یکسانی دارند.")

        producer1_total_cost = v_prime_1 * producer1_unit_cost
        producer2_total_cost = v_prime_2 * producer2_unit_cost

        print(f"هزینه کل تأمین از تولیدکننده 1: {producer1_total_cost:.2f}")
        print(f"هزینه کل تأمین از تولیدکننده 2: {producer2_total_cost:.2f}")
        print(
            f"نسبت هزینه تولیدکننده 1 به کل: {producer1_total_cost / (producer1_total_cost + producer2_total_cost) * 100:.2f}%")
        print(
            f"نسبت هزینه تولیدکننده 2 به کل: {producer2_total_cost / (producer1_total_cost + producer2_total_cost) * 100:.2f}%")

        # تفسیر زمان‌بندی
        print("\n--- تفسیر زمان‌بندی بهینه ---")
        print(f"زمان شروع دوز اول برای گروه 1: {self.tau1[0]}")
        print(f"زمان شروع دوز دوم برای گروه 1: {self.tau2[0]}")
        print(f"فاصله بین دوزها برای گروه 1: {self.tau2[0] - self.tau1[0]} روز")

        print(f"زمان شروع دوز اول برای گروه 2: {self.tau1[1]}")
        print(f"زمان شروع دوز دوم برای گروه 2: {self.tau2[1]}")
        print(f"فاصله بین دوزها برای گروه 2: {self.tau2[1] - self.tau1[1]} روز")

        # اضافه کردن تفسیر عدالت تخصیص
        print("\n--- تفسیر عدالت تخصیص واکسن ---")
        # محاسبه تعداد واکسن تخصیص یافته به هر گروه
        total_susceptible_group1 = sum(self.S[0][t] for t in range(self.tau1[0], self.tau2[0]))
        total_susceptible_group2 = sum(self.S[1][t] for t in range(self.tau1[1], self.tau2[1]))
        total_vaccinated_dose1_group1 = sum(self.V1[0][t] for t in range(self.tau2[0], self.end_time[0] + 1))
        total_vaccinated_dose1_group2 = sum(self.V1[1][t] for t in range(self.tau2[1], self.end_time[1] + 1))

        vaccine_dose1_group1 = value(self.U1[1]) * total_susceptible_group1
        vaccine_dose1_group2 = value(self.U1[2]) * total_susceptible_group2
        total_dose1 = vaccine_dose1_group1 + vaccine_dose1_group2

        if abs(value(self.U1[1]) - value(self.U1[2])) <= 0.1:
            print("توزیع واکسن دوز اول بین دو گروه نسبتاً متعادل است.")
        else:
            print(f"اختلاف نسبت واکسیناسیون دوز اول بین دو گروه: {abs(value(self.U1[1]) - value(self.U1[2])):.4f}")
            if value(self.U1[1]) > value(self.U1[2]):
                print("گروه 1 (افراد بالای 60 سال) نسبت بیشتری از واکسن دوز اول را دریافت می‌کنند.")
            else:
                print("گروه 2 (افراد دارای کسب و کار) نسبت بیشتری از واکسن دوز اول را دریافت می‌کنند.")

        # تحلیل تناسب جمعیتی
        total_pop_group1 = self.S[0][0] + self.I[0][0] + self.Q[0][0] + self.V1[0][0] + self.V2[0][0] + self.R[0][0]
        total_pop_group2 = self.S[1][0] + self.I[1][0] + self.Q[1][0] + self.V1[1][0] + self.V2[1][0] + self.R[1][0]
        total_population = total_pop_group1 + total_pop_group2

        population_ratio_group1 = total_pop_group1 / total_population
        allocation_ratio_dose1 = vaccine_dose1_group1 / total_dose1

        print(f"نسبت جمعیت گروه 1: {population_ratio_group1:.4f} ({population_ratio_group1 * 100:.1f}%)")
        print(f"نسبت تخصیص واکسن دوز اول به گروه 1: {allocation_ratio_dose1:.4f} ({allocation_ratio_dose1 * 100:.1f}%)")

        if allocation_ratio_dose1 >= population_ratio_group1:
            print("گروه 1 (افراد بالای 60 سال) سهم بیشتری از واکسن نسبت به نسبت جمعیتش دریافت می‌کند.")
        else:
            print("گروه 1 (افراد بالای 60 سال) سهم کمتری از واکسن نسبت به نسبت جمعیتش دریافت می‌کند.")
            equity_gap = (population_ratio_group1 - allocation_ratio_dose1) * 100
            print(f"شکاف عدالت توزیع: {equity_gap:.2f}%")

    def create_standard_plots(self, results):
        """
        رسم نمودارهای استاندارد بدون تحلیل حساسیت
        """
        print("در حال رسم نمودارهای استاندارد...")

        # کتابخانه‌های مورد نیاز برای اصلاح متن فارسی
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            def fix_farsi_text(text):
                reshaped_text = arabic_reshaper.reshape(text)
                return get_display(reshaped_text)

            support_farsi = True
        except ImportError:
            def fix_farsi_text(text):
                replacements = {
                    'گروه 1\n(افراد بالای 60 سال)': 'Group 1\n(Elderly, 60+)',
                    'گروه 2\n(افراد دارای کسب و کار)': 'Group 2\n(Business owners)',
                    'دوز اول': 'First dose',
                    'دوز دوم': 'Second dose',
                    'گروه‌های اولویت': 'Priority Groups',
                    'نسبت واکسیناسیون': 'Vaccination Ratio',
                    'نسبت بهینه واکسیناسیون هر گروه': 'Optimal Vaccination Ratio for Each Group'
                }
                return replacements.get(text, text)

            support_farsi = False

        # نمودار اصلی: نسبت واکسیناسیون
        plt.figure(figsize=(10, 6))
        groups = ['گروه 1\n(افراد بالای 60 سال)', 'گروه 2\n(افراد دارای کسب و کار)']
        fixed_groups = [fix_farsi_text(group) for group in groups]

        values_dose1 = [results['U1'][1], results['U1'][2]]
        values_dose2 = [results['U2'][1], results['U2'][2]]

        x = np.arange(len(groups))
        width = 0.35

        bars1 = plt.bar(x - width / 2, values_dose1, width,
                        label=fix_farsi_text('دوز اول'), color='skyblue')
        bars2 = plt.bar(x + width / 2, values_dose2, width,
                        label=fix_farsi_text('دوز دوم'), color='lightgreen')

        plt.xlabel(fix_farsi_text('گروه‌های اولویت'))
        plt.ylabel(fix_farsi_text('نسبت واکسیناسیون'))
        plt.title(fix_farsi_text('نسبت بهینه واکسیناسیون هر گروه'))
        plt.xticks(x, fixed_groups)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # اضافه کردن برچسب‌ها
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                     f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                     f'{height:.2%}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('vaccination_ratio_standard.png')
        plt.close()
        print("✅ نمودار استاندارد ذخیره شد: vaccination_ratio_standard.png")

    def analyze_timing_sensitivity(self):
        """
        تحلیل حساسیت زمان‌بندی و رسم نمودارهای تحلیلی - تصحیح شده
        """
        print("\n=== شروع تحلیل حساسیت زمان‌بندی ===")

        # محدوده‌های زمانی تصحیح شده برای تست
        tau1_range = range(30, 51, 3)  # تصحیح شده: 30, 33, 36, 39, 42, 45, 48
        tau2_base = 80  # زمان پایه تصحیح شده برای دوز دوم
        gap_range = range(45, 76, 5)  # تصحیح شده: فاصله‌های مختلف بین دوزها

        # ذخیره نتایج
        sensitivity_results = {
            'tau1_values': [],
            'total_costs': [],
            'z1_costs': [],
            'z2_costs': [],
            'z3_costs': [],
            'tau1_tau2_matrix': {},
            'gap_analysis': {}
        }

        print("در حال تست زمان‌های مختلف...")

        # تحلیل تأثیر tau1
        for tau1 in tqdm(tau1_range, desc="تحلیل τ1"):
            tau2 = max(tau1 + 45, tau2_base)  # تصحیح شده: حداقل 45 روز فاصله

            try:
                # اجرای مدل با زمان‌های جدید
                temp_tau1 = [tau1, tau1]  # هر دو گروه همزمان شروع
                temp_tau2 = [tau2, tau2 - 5]  # گروه 2 کمی زودتر دوز دوم

                self.build_model(tau1=temp_tau1, tau2=temp_tau2)
                temp_results = self.solve_model()

                if temp_results:
                    sensitivity_results['tau1_values'].append(tau1)
                    sensitivity_results['total_costs'].append(temp_results['objective_value'])
                    sensitivity_results['z1_costs'].append(temp_results['objective1_value'])
                    sensitivity_results['z2_costs'].append(temp_results['objective2_value'])
                    sensitivity_results['z3_costs'].append(temp_results['objective3_value'])

            except Exception as e:
                print(f"خطا در تست τ1={tau1}: {e}")
                continue

        # تحلیل ماتریس tau1-tau2 تصحیح شده
        print("در حال تحلیل ماتریس زمان‌بندی...")
        tau1_test_range = range(30, 46, 3)  # تصحیح شده: 30, 33, 36, 39, 42, 45
        tau2_test_range = range(75, 126, 8)  # تصحیح شده: 75, 83, 91, 99, 107, 115, 123

        cost_matrix = []
        for tau2 in tau2_test_range:
            row = []
            for tau1 in tau1_test_range:
                if tau2 > tau1 + 40:  # تصحیح شده: حداقل فاصله 40 روز
                    try:
                        temp_tau1 = [tau1, tau1]
                        temp_tau2 = [tau2, tau2 - 5]

                        self.build_model(tau1=temp_tau1, tau2=temp_tau2)
                        temp_results = self.solve_model()

                        if temp_results:
                            row.append(temp_results['objective_value'])
                        else:
                            row.append(float('inf'))
                    except:
                        row.append(float('inf'))
                else:
                    row.append(float('inf'))
            cost_matrix.append(row)

        sensitivity_results['tau1_tau2_matrix'] = {
            'tau1_range': list(tau1_test_range),
            'tau2_range': list(tau2_test_range),
            'cost_matrix': cost_matrix
        }

        # بازگردادن به مدل اصلی
        self.build_model()

        return sensitivity_results

    def create_timing_analysis_plots(self, sensitivity_results):
        """
        رسم نمودارهای تحلیل زمان‌بندی - تصحیح شده
        """
        print("در حال رسم نمودارهای تحلیل زمان‌بندی...")

        # تابع کمکی برای متن فارسی
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            def fix_farsi_text(text):
                reshaped_text = arabic_reshaper.reshape(text)
                return get_display(reshaped_text)

            support_farsi = True
        except ImportError:
            def fix_farsi_text(text):
                replacements = {
                    'تحلیل حساسیت: هزینه در برابر زمان شروع': 'Sensitivity Analysis: Cost vs Start Time',
                    'زمان شروع دوز اول (روز)': 'First Dose Start Time (days)',
                    'هزینه کل نرمال‌شده': 'Total Normalized Cost',
                    'نقشه هزینه: بهینه‌ترین زمان‌بندی': 'Cost Map: Optimal Timing',
                    'زمان شروع دوز دوم (روز)': 'Second Dose Start Time (days)',
                    'تحلیل مؤلفه‌های هزینه': 'Cost Components Analysis',
                    'هزینه تأمین واکسن': 'Vaccine Supply Cost',
                    'هزینه‌های اجتماعی': 'Social Costs',
                    'هزینه‌های اقتصادی': 'Economic Costs',
                    'نقطه بهینه فعلی': 'Current Optimal Point'
                }
                return replacements.get(text, text)

            support_farsi = False

        # نمودار 1: هزینه در برابر tau1 - تصحیح شده
        if sensitivity_results['tau1_values'] and sensitivity_results['total_costs']:
            plt.figure(figsize=(12, 6))

            tau1_vals = sensitivity_results['tau1_values']
            costs = sensitivity_results['total_costs']

            plt.plot(tau1_vals, costs, 'b-o', linewidth=2, markersize=8, label='هزینه کل')

            # نشان دادن نقطه بهینه
            min_cost_idx = np.argmin(costs)
            optimal_tau1 = tau1_vals[min_cost_idx]
            optimal_cost = costs[min_cost_idx]

            plt.plot(optimal_tau1, optimal_cost, 'r*', markersize=15,
                     label=f'بهینه: τ1={optimal_tau1}, هزینه={optimal_cost:.2f}')
            plt.axvline(x=optimal_tau1, color='red', linestyle='--', alpha=0.7)

            plt.xlabel(fix_farsi_text('زمان شروع دوز اول (روز)'))
            plt.ylabel(fix_farsi_text('هزینه کل نرمال‌شده'))
            plt.title(fix_farsi_text('تحلیل حساسیت: هزینه در برابر زمان شروع'))
            plt.grid(True, alpha=0.3)
            plt.legend()

            # اضافه کردن برچسب‌ها
            for i, (x, y) in enumerate(zip(tau1_vals, costs)):
                if i % 2 == 0:  # فقط برخی نقاط
                    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                                 xytext=(0, 10), ha='center', fontsize=9)

            plt.tight_layout()
            plt.savefig('timing_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ نمودار تحلیل حساسیت ذخیره شد: timing_sensitivity_analysis.png")

        # نمودار 2: نقشه حرارتی (Heatmap) - تصحیح شده
        if sensitivity_results['tau1_tau2_matrix']['cost_matrix']:
            plt.figure(figsize=(10, 8))

            matrix_data = sensitivity_results['tau1_tau2_matrix']
            cost_matrix = np.array(matrix_data['cost_matrix'])

            # جایگزینی inf با NaN برای نمایش بهتر
            cost_matrix[cost_matrix == float('inf')] = np.nan

            # رسم heatmap
            im = plt.imshow(cost_matrix, cmap='viridis', aspect='auto',
                            interpolation='nearest', origin='lower')

            # تنظیم محورها
            plt.xticks(range(len(matrix_data['tau1_range'])), matrix_data['tau1_range'])
            plt.yticks(range(len(matrix_data['tau2_range'])), matrix_data['tau2_range'])

            plt.xlabel(fix_farsi_text('زمان شروع دوز اول (روز)'))
            plt.ylabel(fix_farsi_text('زمان شروع دوز دوم (روز)'))
            plt.title(fix_farsi_text('نقشه هزینه: بهینه‌ترین زمان‌بندی'))

            # نشان دادن نقطه بهینه
            if not np.all(np.isnan(cost_matrix)):
                min_pos = np.unravel_index(np.nanargmin(cost_matrix), cost_matrix.shape)
                plt.plot(min_pos[1], min_pos[0], 'r*', markersize=20,
                         label=fix_farsi_text('نقطه بهینه فعلی'))

            # colorbar
            cbar = plt.colorbar(im, label=fix_farsi_text('هزینه کل نرمال‌شده'))
            plt.legend()

            plt.tight_layout()
            plt.savefig('timing_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ نقشه حرارتی زمان‌بندی ذخیره شد: timing_heatmap.png")

        # نمودار 3: تحلیل مؤلفه‌های هزینه
        if (sensitivity_results['tau1_values'] and
                sensitivity_results['z1_costs'] and
                sensitivity_results['z2_costs'] and
                sensitivity_results['z3_costs']):

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            tau1_vals = sensitivity_results['tau1_values']

            # Z1: هزینه تأمین
            ax1.plot(tau1_vals, sensitivity_results['z1_costs'], 'b-o', linewidth=2)
            ax1.set_title(fix_farsi_text('هزینه تأمین واکسن'))
            ax1.set_xlabel('τ1')
            ax1.grid(True, alpha=0.3)

            # Z2: هزینه اجتماعی
            ax2.plot(tau1_vals, sensitivity_results['z2_costs'], 'g-o', linewidth=2)
            ax2.set_title(fix_farsi_text('هزینه‌های اجتماعی'))
            ax2.set_xlabel('τ1')
            ax2.grid(True, alpha=0.3)

            # Z3: هزینه اقتصادی
            ax3.plot(tau1_vals, sensitivity_results['z3_costs'], 'orange',
                     marker='o', linewidth=2)
            ax3.set_title(fix_farsi_text('هزینه‌های اقتصادی'))
            ax3.set_xlabel('τ1')
            ax3.grid(True, alpha=0.3)

            # هزینه کل
            ax4.plot(tau1_vals, sensitivity_results['total_costs'], 'r-o', linewidth=2)
            ax4.set_title('هزینه کل (نرمال‌شده)')
            ax4.set_xlabel('τ1')
            ax4.grid(True, alpha=0.3)

            # نشان دادن نقاط بهینه
            for ax, costs in zip([ax1, ax2, ax3, ax4],
                                 [sensitivity_results['z1_costs'],
                                  sensitivity_results['z2_costs'],
                                  sensitivity_results['z3_costs'],
                                  sensitivity_results['total_costs']]):
                min_idx = np.argmin(costs)
                optimal_tau = tau1_vals[min_idx]
                ax.axvline(x=optimal_tau, color='red', linestyle='--', alpha=0.5)

            plt.suptitle(fix_farsi_text('تحلیل مؤلفه‌های هزینه'), fontsize=16)
            plt.tight_layout()
            plt.savefig('cost_components_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ تحلیل مؤلفه‌های هزینه ذخیره شد: cost_components_analysis.png")

        # نمودار 4: نمودار مقایسه‌ای - تصحیح شده
        if sensitivity_results['tau1_values'] and sensitivity_results['total_costs']:
            plt.figure(figsize=(12, 8))

            tau1_vals = sensitivity_results['tau1_values']
            costs = sensitivity_results['total_costs']

            # رنگ‌بندی بر اساس کیفیت
            colors = []
            min_cost = min(costs)
            max_cost = max(costs)

            for cost in costs:
                if cost <= min_cost + 0.1 * (max_cost - min_cost):
                    colors.append('green')  # بهینه
                elif cost <= min_cost + 0.3 * (max_cost - min_cost):
                    colors.append('yellow')  # قابل قبول
                else:
                    colors.append('red')  # ضعیف

            bars = plt.bar(tau1_vals, costs, color=colors, alpha=0.7, edgecolor='black')

            # اضافه کردن برچسب‌ها
            for bar, cost, tau1 in zip(bars, costs, tau1_vals):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{cost:.2f}', ha='center', va='bottom', fontweight='bold')

            plt.xlabel(fix_farsi_text('زمان شروع دوز اول (روز)'))
            plt.ylabel(fix_farsi_text('هزینه کل نرمال‌شده'))
            plt.title('مقایسه کیفیت زمان‌بندی‌های مختلف')

            # افزودن خط مرجع
            plt.axhline(y=min_cost, color='green', linestyle='-', alpha=0.5,
                        label=f'کمترین هزینه: {min_cost:.2f}')
            plt.axhline(y=min_cost + 0.1 * (max_cost - min_cost), color='yellow',
                        linestyle='--', alpha=0.5, label='حد قابل قبول')

            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('timing_quality_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ نمودار مقایسه کیفیت ذخیره شد: timing_quality_comparison.png")

    def visualize_results(self, results):
        """
        رسم نمودارهای نتایج با متن فارسی صحیح + تحلیل زمان‌بندی تصحیح شده
        """
        if results is None:
            print("نتایج برای رسم نمودار موجود نیست.")
            return

        print("\nدر حال رسم نمودارهای نتایج...")

        # اطمینان از بستن نمودارهای قبلی
        plt.close('all')

        # تحلیل حساسیت زمان‌بندی تصحیح شده
        print("\n🎯 شروع تحلیل حساسیت زمان‌بندی تصحیح شده...")
        sensitivity_results = self.analyze_timing_sensitivity()
        self.create_timing_analysis_plots(sensitivity_results)

        # کتابخانه‌های مورد نیاز برای اصلاح متن فارسی
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display

            # تابع کمکی برای اصلاح متن فارسی
            def fix_farsi_text(text):
                """اصلاح متن فارسی برای نمایش صحیح در matplotlib"""
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                return bidi_text

            support_farsi = True
            print("پشتیبانی از متن فارسی فعال شد.")

        except ImportError:
            # اگر کتابخانه‌ها نصب نشده باشند، از متن انگلیسی استفاده می‌کنیم
            def fix_farsi_text(text):
                # جایگزینی متن‌های فارسی با معادل انگلیسی
                replacements = {
                    'گروه 1\n(افراد بالای 60 سال)': 'Group 1\n(Elderly, 60+)',
                    'گروه 2\n(افراد دارای کسب و کار)': 'Group 2\n(Business owners)',
                    'دوز اول': 'First dose',
                    'دوز دوم': 'Second dose',
                    'گروه‌های اولویت': 'Priority Groups',
                    'نسبت واکسیناسیون': 'Vaccination Ratio',
                    'نسبت بهینه واکسیناسیون هر گروه': 'Optimal Vaccination Ratio for Each Group',
                    'تولیدکننده 1': 'Manufacturer 1',
                    'تولیدکننده 2': 'Manufacturer 2',
                    'تولیدکنندگان': 'Manufacturers',
                    'تعداد واکسن': 'Number of Vaccines',
                    'تعداد بهینه واکسن تولید شده توسط هر تولیدکننده': 'Optimal Number of Vaccines Produced by Each Manufacturer',
                    'توزیع مصرف واکسن': 'Vaccine Usage Distribution',
                    'ظرفیت باقیمانده': 'Remaining Capacity',
                    'توزیع تولید واکسن و ظرفیت باقیمانده': 'Vaccine Production Distribution and Remaining Capacity',
                    'زمان‌بندی واکسیناسیون': 'Vaccination Schedule',
                    'زمان (روز)': 'Time (days)',
                    'زمان‌بندی بهینه واکسیناسیون': 'Optimal Vaccination Schedule',
                    'هزینه واحد': 'Unit Cost',
                    'مقایسه تولیدکنندگان': 'Manufacturers Comparison',
                    'عدالت تخصیص واکسن': 'Vaccine Allocation Equity',
                    'نسبت به جمعیت': 'Population Ratio',
                    'نسبت به تخصیص': 'Allocation Ratio'
                }
                return replacements.get(text, text)

            support_farsi = False
            print("هشدار: کتابخانه‌های arabic_reshaper و python-bidi نصب نشده‌اند.")
            print("برای نمایش صحیح متن فارسی، لطفاً این کتابخانه‌ها را نصب کنید:")
            print("pip install arabic_reshaper python-bidi")
            print("از متن انگلیسی به جای فارسی استفاده می‌شود.")

        # نمودار 1: نسبت واکسیناسیون هر گروه
        plt.figure(figsize=(10, 6))
        groups = ['گروه 1\n(افراد بالای 60 سال)', 'گروه 2\n(افراد دارای کسب و کار)']
        fixed_groups = [fix_farsi_text(group) for group in groups]

        # مقادیر
        values_dose1 = [results['U1'][1], results['U1'][2]]
        values_dose2 = [results['U2'][1], results['U2'][2]]

        x = np.arange(len(groups))
        width = 0.35

        bars1 = plt.bar(x - width / 2, values_dose1, width,
                        label=fix_farsi_text('دوز اول'), color='skyblue')
        bars2 = plt.bar(x + width / 2, values_dose2, width,
                        label=fix_farsi_text('دوز دوم'), color='lightgreen')

        plt.xlabel(fix_farsi_text('گروه‌های اولویت'))
        plt.ylabel(fix_farsi_text('نسبت واکسیناسیون'))
        plt.title(fix_farsi_text('نسبت بهینه واکسیناسیون هر گروه'))
        plt.xticks(x, fixed_groups)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # اضافه کردن برچسب روی ستون‌ها
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                         f'{height:.2%}', ha='center', va='bottom', fontsize=10)

        add_labels(bars1)
        add_labels(bars2)

        plt.tight_layout()
        plt.savefig('vaccination_ratio.png')
        plt.close()  # بستن نمودار برای آزادسازی حافظه
        print("نمودار نسبت واکسیناسیون ذخیره شد: vaccination_ratio.png")

        # نمودار 2: تعداد واکسن تولید شده توسط هر تولیدکننده
        plt.figure(figsize=(10, 8))

        # زیرنمودار 1: تعداد واکسن تولید شده
        plt.subplot(2, 1, 1)
        manufacturers = ['تولیدکننده 1', 'تولیدکننده 2']
        fixed_manufacturers = [fix_farsi_text(m) for m in manufacturers]
        values = [results['V_prime'][1], results['V_prime'][2]]

        bars = plt.bar(fixed_manufacturers, values, color=['skyblue', 'lightgreen'])
        plt.xlabel(fix_farsi_text('تولیدکنندگان'))
        plt.ylabel(fix_farsi_text('تعداد واکسن'))
        plt.title(fix_farsi_text('تعداد بهینه واکسن تولید شده توسط هر تولیدکننده'))
        plt.grid(True, alpha=0.3)

        # اضافه کردن برچسب روی ستون‌ها
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 5,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=10)

        # زیرنمودار 2: مقایسه هزینه واحد تولیدکنندگان
        plt.subplot(2, 1, 2)
        unit_costs = [self.P[0], self.P[1]]

        bars_cost = plt.bar(fixed_manufacturers, unit_costs, color=['coral', 'lightseagreen'])
        plt.xlabel(fix_farsi_text('تولیدکنندگان'))
        plt.ylabel(fix_farsi_text('هزینه واحد'))
        plt.title(fix_farsi_text('مقایسه هزینه واحد تولیدکنندگان'))
        plt.grid(True, alpha=0.3)

        # اضافه کردن برچسب روی ستون‌ها
        for bar in bars_cost:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('vaccine_production.png')
        plt.close()  # بستن نمودار برای آزادسازی حافظه
        print("نمودار تولید واکسن ذخیره شد: vaccine_production.png")

        # نمودار 3: مقایسه تولید و مصرف واکسن
        total_vax_need = 0
        dose1_needs = []
        dose2_needs = []

        for j in range(1, self.num_groups + 1):
            j_idx = j - 1  # اندیس آرایه

            # مجموع افراد مستعد بین زمان شروع دوز اول و دوز دوم
            total_susceptible = sum(self.S[j_idx][t] for t in range(self.tau1[j_idx], self.tau2[j_idx]))

            # مجموع افراد واکسینه شده دوز اول بین زمان شروع دوز دوم و پایان اپیدمی
            total_vaccinated_dose1 = sum(self.V1[j_idx][t] for t in range(self.tau2[j_idx], self.end_time[j_idx] + 1))

            # واکسن مورد نیاز برای این گروه
            dose1_need = results['U1'][j] * total_susceptible
            dose2_need = results['U2'][j] * total_vaccinated_dose1

            dose1_needs.append(dose1_need)
            dose2_needs.append(dose2_need)

            total_vax_need += dose1_need + dose2_need

        # کل واکسن تولید شده
        total_production = sum(results['V_prime'][i] for i in range(1, self.num_manufacturers + 1))

        plt.figure(figsize=(12, 6))

        # نمودار میله‌ای برای مصرف واکسن
        plt.subplot(1, 2, 1)
        x = np.arange(len(groups))
        width = 0.35
        plt.bar(x - width / 2, dose1_needs, width, label=fix_farsi_text('دوز اول'))
        plt.bar(x - width / 2, dose2_needs, width, bottom=dose1_needs, label=fix_farsi_text('دوز دوم'))
        plt.xlabel(fix_farsi_text('گروه‌های اولویت'))
        plt.ylabel(fix_farsi_text('تعداد واکسن'))
        plt.title(fix_farsi_text('توزیع مصرف واکسن'))
        plt.xticks(x, fixed_groups)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # نمودار دایره‌ای برای تولید واکسن
        plt.subplot(1, 2, 2)
        pie_labels = [fix_farsi_text('تولیدکننده 1'),
                      fix_farsi_text('تولیدکننده 2'),
                      fix_farsi_text('ظرفیت باقیمانده')]

        plt.pie([results['V_prime'][1], results['V_prime'][2], self.L - total_production],
                labels=pie_labels,
                autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightgray'])
        plt.axis('equal')
        plt.title(fix_farsi_text('توزیع تولید واکسن و ظرفیت باقیمانده'))

        plt.tight_layout()
        plt.savefig('vaccine_supply_demand.png')
        plt.close()  # بستن نمودار برای آزادسازی حافظه
        print("نمودار تولید و مصرف واکسن ذخیره شد: vaccine_supply_demand.png")

        # نمودار 4: زمان‌بندی واکسیناسیون (نمودار اصلی) - تصحیح شده
        plt.figure(figsize=(10, 6))

        # محور افقی: روزهای اپیدمی
        days = np.arange(1, self.T + 1)

        # ایجاد خطوط عمودی برای نشان دادن زمان‌های شروع واکسیناسیون - تصحیح شده
        plt.axvline(x=self.tau1[0], color='blue', linestyle='-', alpha=0.5, label=f"τ1_1: {self.tau1[0]}")
        plt.axvline(x=self.tau2[0], color='blue', linestyle='--', alpha=0.5, label=f"τ2_1: {self.tau2[0]}")
        plt.axvline(x=self.tau1[1], color='green', linestyle='-', alpha=0.5, label=f"τ1_2: {self.tau1[1]}")
        plt.axvline(x=self.tau2[1], color='green', linestyle='--', alpha=0.5, label=f"τ2_2: {self.tau2[1]}")

        # رسم منحنی‌های اپیدمی برای هر دو گروه
        plt.plot(days, self.I[0], 'b-', alpha=0.7, label='موارد آلوده گروه 1')
        plt.plot(days, self.I[1], 'g-', alpha=0.7, label='موارد آلوده گروه 2')

        plt.xlabel(fix_farsi_text('زمان (روز)'))
        plt.ylabel('تعداد موارد')
        plt.title(fix_farsi_text('زمان‌بندی بهینه واکسیناسیون'))
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('vaccination_schedule.png')
        plt.close()  # بستن نمودار برای آزادسازی حافظه
        print("نمودار زمان‌بندی واکسیناسیون ذخیره شد: vaccination_schedule.png")

        # نمودار 5: عدالت تخصیص واکسن
        plt.figure(figsize=(10, 6))

        # محاسبه مقادیر برای نمودار
        total_susceptible_group1 = sum(self.S[0][t] for t in range(self.tau1[0], self.tau2[0]))
        total_susceptible_group2 = sum(self.S[1][t] for t in range(self.tau1[1], self.tau2[1]))
        total_vaccinated_dose1_group1 = sum(self.V1[0][t] for t in range(self.tau2[0], self.end_time[0] + 1))
        total_vaccinated_dose1_group2 = sum(self.V1[1][t] for t in range(self.tau2[1], self.end_time[1] + 1))

        # محاسبه تعداد و درصد واکسن تخصیص یافته
        vaccine_dose1_group1 = results['U1'][1] * total_susceptible_group1
        vaccine_dose1_group2 = results['U1'][2] * total_susceptible_group2
        vaccine_dose2_group1 = results['U2'][1] * total_vaccinated_dose1_group1
        vaccine_dose2_group2 = results['U2'][2] * total_vaccinated_dose1_group2

        total_dose1 = vaccine_dose1_group1 + vaccine_dose1_group2
        total_dose2 = vaccine_dose2_group1 + vaccine_dose2_group2

        # محاسبه نسبت جمعیت و نسبت تخصیص
        total_pop_group1 = self.S[0][0] + self.I[0][0] + self.Q[0][0] + self.V1[0][0] + self.V2[0][0] + self.R[0][0]
        total_pop_group2 = self.S[1][0] + self.I[1][0] + self.Q[1][0] + self.V1[1][0] + self.V2[1][0] + self.R[1][0]
        total_population = total_pop_group1 + total_pop_group2

        population_ratio_group1 = total_pop_group1 / total_population
        population_ratio_group2 = total_pop_group2 / total_population

        allocation_ratio_dose1_group1 = vaccine_dose1_group1 / total_dose1
        allocation_ratio_dose1_group2 = vaccine_dose1_group2 / total_dose1

        # نمودار مقایسه نسبت جمعیت و نسبت تخصیص
        categories = ['گروه 1\n(افراد بالای 60 سال)', 'گروه 2\n(افراد دارای کسب و کار)']
        fixed_categories = [fix_farsi_text(cat) for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width / 2, [population_ratio_group1, population_ratio_group2], width,
                       label=fix_farsi_text('نسبت به جمعیت'), color='skyblue')
        bars2 = ax.bar(x + width / 2, [allocation_ratio_dose1_group1, allocation_ratio_dose1_group2], width,
                       label=fix_farsi_text('نسبت به تخصیص'), color='lightgreen')

        ax.set_xlabel(fix_farsi_text('گروه‌های اولویت'))
        ax.set_ylabel('درصد')
        ax.set_title(fix_farsi_text('عدالت تخصیص واکسن'))
        ax.set_xticks(x)
        ax.set_xticklabels(fixed_categories)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # افزودن برچسب درصد روی نمودار
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom')

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('vaccine_equity.png')
        plt.close()  # بستن نمودار برای آزادسازی حافظه
        print("نمودار عدالت تخصیص واکسن ذخیره شد: vaccine_equity.png")

        print("\n🎯 === خلاصه نمودارهای ایجاد شده ===")
        print("✅ نمودارهای اصلی:")
        print("   - vaccination_ratio.png")
        print("   - vaccine_production.png")
        print("   - vaccine_supply_demand.png")
        print("   - vaccination_schedule.png")
        print("   - vaccine_equity.png")
        print("\n🚀 نمودارهای تحلیل زمان‌بندی (تصحیح شده):")
        print("   - timing_sensitivity_analysis.png")
        print("   - timing_heatmap.png")
        print("   - cost_components_analysis.png")
        print("   - timing_quality_comparison.png")
        print("\n💡 این نمودارها نشان می‌دهند:")
        print("   🎯 کدام زمان‌ها بهینه‌اند و چرا (محدوده 30-50 روز)")
        print("   📊 حساسیت هزینه نسبت به تغییرات زمان")
        print("   🔍 مقایسه کیفی زمان‌های مختلف")
        print("   🗺️ نقشه کامل فضای جستجو")

    def find_optimal_timing(self):
        """
        جستجوی ترکیب بهینه زمان‌های واکسیناسیون - تصحیح شده
        """
        if self.config is None:
            print("خطا: فایل پیکربندی بارگذاری نشده است. ابتدا فایل config.json را بارگذاری کنید.")
            return None

        print("\n=== شروع جستجوی زمان‌های بهینه واکسیناسیون ===")

        # استخراج محدوده‌های زمانی تصحیح شده از پیکربندی
        tau1_1_min = self.config['tau1_group1_min']
        tau1_1_max = self.config['tau1_group1_max']

        tau1_2_min = self.config['tau1_group2_min']
        tau1_2_max = self.config['tau1_group2_max']

        # ایجاد محدوده‌های زمانی با استفاده از مقادیر تصحیح شده
        tau1_1_range = range(tau1_1_min,
                             tau1_1_max + 1,
                             self.config['time_step'])

        tau1_2_range = range(tau1_2_min,
                             tau1_2_max + 1,
                             self.config['time_step'])

        # متغیرهای نگهداری بهترین نتیجه
        best_cost = float('inf')
        best_timing = None
        best_results = None

        # تعداد کل حالت‌های ممکن
        total_combinations = 0
        for tau1_1 in tau1_1_range:
            tau2_1_min = tau1_1 + self.config['gap_group1_min']
            tau2_1_max = min(tau1_1 + self.config['gap_group1_max'], self.T - 1)

            for tau2_1 in range(tau2_1_min, tau2_1_max + 1, self.config['time_step']):
                for tau1_2 in tau1_2_range:
                    tau2_2_min = tau1_2 + self.config['gap_group2_min']
                    tau2_2_max = min(tau1_2 + self.config['gap_group2_max'], self.T - 1)

                    for tau2_2 in range(tau2_2_min, tau2_2_max + 1, self.config['time_step']):
                        if tau2_1 < self.T and tau2_2 < self.T:
                            total_combinations += 1

        print(f"تعداد کل ترکیب‌های ممکن زمان‌بندی: {total_combinations}")

        # ایجاد نوار پیشرفت
        with tqdm(total=total_combinations, desc="پیشرفت جستجو") as pbar:
            # جستجوی تمام ترکیب‌های ممکن
            for tau1_1 in tau1_1_range:
                tau2_1_min = tau1_1 + self.config['gap_group1_min']
                tau2_1_max = min(tau1_1 + self.config['gap_group1_max'], self.T - 1)

                for tau2_1 in range(tau2_1_min, tau2_1_max + 1, self.config['time_step']):
                    for tau1_2 in tau1_2_range:
                        tau2_2_min = tau1_2 + self.config['gap_group2_min']
                        tau2_2_max = min(tau1_2 + self.config['gap_group2_max'], self.T - 1)

                        for tau2_2 in range(tau2_2_min, tau2_2_max + 1, self.config['time_step']):
                            if tau2_1 >= self.T or tau2_2 >= self.T:
                                pbar.update(1)
                                continue

                            # ترکیب زمانی فعلی
                            current_tau1 = [tau1_1, tau1_2]
                            current_tau2 = [tau2_1, tau2_2]

                            # ساخت و حل مدل با این زمان‌ها
                            self.build_model(tau1=current_tau1, tau2=current_tau2)
                            results = self.solve_model()

                            # بررسی نتایج
                            if results and results['objective_value'] < best_cost:
                                best_cost = results['objective_value']
                                best_timing = {
                                    'tau1_1': tau1_1,
                                    'tau2_1': tau2_1,
                                    'tau1_2': tau1_2,
                                    'tau2_2': tau2_2
                                }
                                best_results = results

                            pbar.update(1)

        if best_timing:
            print("\n=== زمان‌های بهینه یافت شده ===")
            print(f"زمان شروع دوز اول برای گروه 1 (τ1_1): {best_timing['tau1_1']}")
            print(f"زمان شروع دوز دوم برای گروه 1 (τ2_1): {best_timing['tau2_1']}")
            print(f"فاصله بین دوزها برای گروه 1: {best_timing['tau2_1'] - best_timing['tau1_1']} روز")

            print(f"زمان شروع دوز اول برای گروه 2 (τ1_2): {best_timing['tau1_2']}")
            print(f"زمان شروع دوز دوم برای گروه 2 (τ2_2): {best_timing['tau2_2']}")
            print(f"فاصله بین دوزها برای گروه 2: {best_timing['tau2_2'] - best_timing['tau1_2']} روز")

            print(f"هزینه کل با این زمان‌بندی: {best_cost:.2f}")

            # ذخیره نتایج در یک فایل JSON
            output = {
                'optimal_timing': best_timing,
                'optimal_cost': best_cost,
                'allocation_results': {
                    'U1': {str(k): v for k, v in best_results['U1'].items()},
                    'U2': {str(k): v for k, v in best_results['U2'].items()},
                    'V_prime': {str(k): v for k, v in best_results['V_prime'].items()}
                },
                'equity_metrics': best_results.get('equity_metrics', {})
            }

            with open('optimal_results.json', 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=4)

            print("نتایج بهینه در فایل optimal_results.json ذخیره شد.")

            return best_timing, best_results
        else:
            print("هیچ زمان‌بندی بهینه‌ای یافت نشد.")
            return None

    def run_with_optimal_timing(self):
        """
        اجرای مدل با استفاده از زمان‌های بهینه
        """
        print("\n=== اجرای مدل با زمان‌های بهینه ===")

        # جستجوی زمان‌های بهینه
        optimal_result = self.find_optimal_timing()

        if optimal_result is not None:
            optimal_timing, _ = optimal_result

            # تنظیم زمان‌های بهینه
            tau1 = [optimal_timing['tau1_1'], optimal_timing['tau1_2']]
            tau2 = [optimal_timing['tau2_1'], optimal_timing['tau2_2']]

            # اجرای مدل با زمان‌های بهینه
            self.build_model(tau1=tau1, tau2=tau2)
            results = self.solve_model()

            if results:
                # محاسبه اطلاعات تکمیلی
                self.calculate_additional_info()

                # رسم نمودارها
                self.visualize_results(results)

                # ذخیره نتایج در فایل JSON
                self.save_results_to_json(results, "optimal_results.json")

                # نمایش خلاصه نتایج نهایی با زمان‌های بهینه
                print("\n=== خلاصه نتایج نهایی با زمان‌های بهینه ===")
                print(f"1. زمان شروع دوز اول برای گروه 1: {tau1[0]}")
                print(f"2. زمان شروع دوز دوم برای گروه 1: {tau2[0]}")
                print(f"3. زمان شروع دوز اول برای گروه 2: {tau1[1]}")
                print(f"4. زمان شروع دوز دوم برای گروه 2: {tau2[1]}")
                print(f"5. درصد واکسن دز اول مورد نیاز به گروه اول (افراد بالای 60 سال): {results['U1'][1] * 100:.2f}%")
                print(
                    f"6. درصد واکسن دز اول مورد نیاز به گروه دوم (افراد دارای کسب و کار): {results['U1'][2] * 100:.2f}%")
                print(f"7. درصد واکسن دز دوم مورد نیاز به گروه اول (افراد بالای 60 سال): {results['U2'][1] * 100:.2f}%")
                print(
                    f"8. درصد واکسن دز دوم مورد نیاز به گروه دوم (افراد دارای کسب و کار): {results['U2'][2] * 100:.2f}%")
                print(f"9. تعداد واکسن تولید شده توسط تولید کننده اول: {results['V_prime'][1]:.2f}")
                print(f"10. تعداد واکسن تولید شده توسط تولید کننده دوم: {results['V_prime'][2]:.2f}")
                print(f"11. هزینه کل: {results['objective_value']:.2f}")

                # اضافه کردن اطلاعات عدالت تخصیص
                if 'equity_metrics' in results:
                    metrics = results['equity_metrics']
                    print("\n=== شاخص‌های عدالت تخصیص واکسن ===")
                    print(
                        f"12. شاخص عدالت (اختلاف نسبت واکسیناسیون دوز اول): {metrics.get('equity_diff_dose1', 0):.4f}")
                    print(f"13. کارایی توزیع نسبت به جمعیت: {metrics.get('population_effectiveness', 0):.4f}")

                return results
            else:
                print("مدل با زمان‌های بهینه به جواب بهینه نرسید.")
                return None
        else:
            print("زمان‌بندی بهینه یافت نشد. از زمان‌های پیش‌فرض استفاده می‌شود.")
            # اجرای مدل با زمان‌های پیش‌فرض
            self.build_model()
            results = self.solve_model()

            if results:
                # محاسبه اطلاعات تکمیلی
                self.calculate_additional_info()

                # رسم نمودارها
                self.visualize_results(results)

                # ذخیره نتایج در فایل JSON
                self.save_results_to_json(results, "default_timing_results.json")

                print("\n=== نتایج با زمان‌های پیش‌فرض ===")
                print(f"1. درصد واکسن دز اول مورد نیاز به گروه اول (افراد بالای 60 سال): {results['U1'][1] * 100:.2f}%")
                print(
                    f"2. درصد واکسن دز اول مورد نیاز به گروه دوم (افراد دارای کسب و کار): {results['U1'][2] * 100:.2f}%")
                print(f"3. درصد واکسن دز دوم مورد نیاز به گروه اول (افراد بالای 60 سال): {results['U2'][1] * 100:.2f}%")
                print(
                    f"4. درصد واکسن دز دوم مورد نیاز به گروه دوم (افراد دارای کسب و کار): {results['U2'][2] * 100:.2f}%")
                print(f"5. تعداد واکسن تولید شده توسط تولید کننده اول: {results['V_prime'][1]:.2f}")
                print(f"6. تعداد واکسن تولید شده توسط تولید کننده دوم: {results['V_prime'][2]:.2f}")
                print(f"7. هزینه کل: {results['objective_value']:.2f}")

                return results
            else:
                print("مدل حتی با زمان‌های پیش‌فرض نیز به جواب نرسید.")
                return None

    def save_results_to_json(self, results, filename):
        """
        ذخیره نتایج در فایل JSON

        پارامترها:
            results (dict): نتایج بهینه‌سازی
            filename (str): نام فایل خروجی
        """
        if results:
            # تبدیل نتایج به فرمت قابل ذخیره در JSON
            output = {
                'optimal_timing': {
                    'tau1_1': self.tau1[0],
                    'tau2_1': self.tau2[0],
                    'tau1_2': self.tau1[1],
                    'tau2_2': self.tau2[1]
                },
                'optimal_cost': results['objective_value'],
                'objective_components': {
                    'Z1_supply_cost': results['objective1_value'],
                    'Z2_social_cost': results['objective2_value'],
                    'Z3_economic_cost': results['objective3_value']
                },
                'weights': results['weights'],
                'allocation_results': {
                    'U1': {str(k): v for k, v in results['U1'].items()},
                    'U2': {str(k): v for k, v in results['U2'].items()},
                    'V_prime': {str(k): v for k, v in results['V_prime'].items()}
                },
                'equity_metrics': results.get('equity_metrics', {})
            }

            # ذخیره در فایل JSON
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=4)

            print(f"نتایج در فایل {filename} ذخیره شدند.")

    def run(self, find_optimal_timing=False):
        """
        اجرای کامل فرآیند بهینه‌سازی

        پارامترها:
            find_optimal_timing (bool): آیا زمان‌های بهینه واکسیناسیون جستجو شود؟
        """
        if find_optimal_timing and self.config is not None:
            # اجرای مدل با جستجوی زمان‌های بهینه
            return self.run_with_optimal_timing()
        else:
            # اجرای معمولی مدل
            self.build_model()
            results = self.solve_model()

            # محاسبه اطلاعات تکمیلی و رسم نمودارها
            if results:
                self.calculate_additional_info()

                # انتخاب نوع تحلیل
                timing_analysis = input("\nآیا می‌خواهید تحلیل حساسیت زمان‌بندی انجام شود؟ (بله/خیر): ").strip().lower()

                if timing_analysis in ['بله', 'yes', 'y', '1']:
                    print("🎯 تحلیل کامل با بررسی حساسیت زمان‌بندی...")
                    self.visualize_results(results)
                else:
                    print("📊 رسم نمودارهای استاندارد...")
                    # فقط نمودارهای اصلی
                    self.create_standard_plots(results)

                # ذخیره نتایج
                self.save_results_to_json(results, "results.json")

                # نمایش خلاصه نتایج
                print("\n=== خلاصه نتایج نهایی ===")
                print(f"1. درصد واکسن دز اول مورد نیاز به گروه اول (افراد بالای 60 سال): {results['U1'][1] * 100:.2f}%")
                print(
                    f"2. درصد واکسن دز اول مورد نیاز به گروه دوم (افراد دارای کسب و کار): {results['U1'][2] * 100:.2f}%")
                print(f"3. درصد واکسن دز دوم مورد نیاز به گروه اول (افراد بالای 60 سال): {results['U2'][1] * 100:.2f}%")
                print(
                    f"4. درصد واکسن دز دوم مورد نیاز به گروه دوم (افراد دارای کسب و کار): {results['U2'][2] * 100:.2f}%")
                print(f"5. تعداد واکسن تولید شده توسط تولید کننده اول: {results['V_prime'][1]:.2f}")
                print(f"6. تعداد واکسن تولید شده توسط تولید کننده دوم: {results['V_prime'][2]:.2f}")
                print(f"7. هزینه کل: {results['objective_value']:.2f}")

                # اضافه کردن اطلاعات عدالت تخصیص
                if 'equity_metrics' in results:
                    metrics = results['equity_metrics']
                    print("\n=== شاخص‌های عدالت تخصیص واکسن ===")
                    print(f"8. شاخص عدالت (اختلاف نسبت واکسیناسیون دوز اول): {metrics.get('equity_diff_dose1', 0):.4f}")
                    print(f"9. کارایی توزیع نسبت به جمعیت: {metrics.get('population_effectiveness', 0):.4f}")

                return results

            return None


def main():
    """
    تابع اصلی
    """
    try:
        # بررسی و نصب کتابخانه‌های مورد نیاز
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            print("کتابخانه‌های مورد نیاز برای نمایش متن فارسی نصب شده‌اند.")
        except ImportError:
            print("در حال نصب کتابخانه‌های مورد نیاز برای نمایش متن فارسی...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'arabic_reshaper', 'python-bidi'])
            print("کتابخانه‌های مورد نیاز با موفقیت نصب شدند.")

        try:
            from tqdm import tqdm
            print("کتابخانه نوار پیشرفت (tqdm) نصب شده است.")
        except ImportError:
            print("در حال نصب کتابخانه نوار پیشرفت (tqdm)...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'tqdm'])
            from tqdm import tqdm
            print("کتابخانه نوار پیشرفت با موفقیت نصب شد.")

        # مسیر فایل اکسل - مسیر را به صورت ثابت تعریف می‌کنیم
        excel_file = "Group.xlsx"  # فایل باید در همان پوشه‌ای باشد که اسکریپت اجرا می‌شود
        print(f"استفاده از فایل اکسل: {excel_file}")

        # پرسش در مورد استفاده از فایل پیکربندی
        use_config = input("آیا می‌خواهید از فایل پیکربندی استفاده کنید؟ (بله/خیر): ").strip().lower()

        config_file = None
        if use_config in ['بله', 'yes', 'y', '1']:
            # درخواست مسیر فایل پیکربندی
            config_file = input("لطفاً مسیر دقیق فایل config.json را وارد کنید: ")

            # بررسی وجود فایل پیکربندی
            if not os.path.exists(config_file):
                print(f"خطا: فایل پیکربندی {config_file} یافت نشد.")
                print("ادامه با تنظیمات پیش‌فرض...")
                config_file = None

        # پرسش در مورد استفاده از فایل وزن‌ها
        use_weights = input("آیا می‌خواهید از فایل وزن‌های سفارشی استفاده کنید؟ (بله/خیر): ").strip().lower()

        weights_file_path = None
        if use_weights in ['بله', 'yes', 'y', '1']:
            # درخواست مسیر فایل وزن‌ها
            weights_file_path = input("لطفاً مسیر دقیق فایل weights.json را وارد کنید: ")

            # بررسی وجود فایل وزن‌ها
            if not os.path.exists(weights_file_path):
                print(f"خطا: فایل وزن‌ها {weights_file_path} یافت نشد.")
                print("ادامه با وزن‌های پیش‌فرض...")
                weights_file_path = None

        # بررسی وجود فایل اکسل
        if not os.path.exists(excel_file):
            print(f"خطا: فایل {excel_file} یافت نشد.")
            print(f"لطفاً فایل اکسل خود را با نام '{excel_file}' در همان مسیری که برنامه اجرا می‌شود قرار دهید.")
            return

        # ایجاد و اجرای بهینه‌ساز
        optimizer = VaccineAllocationOptimizer(excel_file, config_file, weights_file_path)

        # پرسش در مورد اجرای جستجوی زمان بهینه
        find_optimal = False
        if config_file:
            find_optimal_input = input("آیا می‌خواهید جستجوی زمان‌های بهینه انجام شود؟ (بله/خیر): ").strip().lower()
            find_optimal = find_optimal_input in ['بله', 'yes', 'y', '1']

        # پرسش در مورد اجرای متوالی با وزن‌های مختلف
        run_multiple = input("آیا می‌خواهید الگوریتم را با چند وزن مختلف اجرا کنید؟ (بله/خیر): ").strip().lower()

        if run_multiple in ['بله', 'yes', 'y', '1']:
            # بارگذاری مجموعه وزن‌ها - یا از فایل یا مقادیر پیش‌فرض
            weight_sets = load_multiple_weights(weights_file_path if use_weights in ['بله', 'yes', 'y', '1'] else None)

            print(f"\n🎯 برنامه با {len(weight_sets)} مجموعه وزن مختلف اجرا خواهد شد:")
            print("محدودیت‌های انعطاف‌پذیر فعال است - انتظار تفاوت‌های واضح را داشته باشید! ✅")
            for idx, weight_set in enumerate(weight_sets):
                print(
                    f"{idx + 1}. {weight_set['name']} (w1={weight_set['w1']}, w2={weight_set['w2']}, w3={weight_set['w3']})")

            results_collection = []

            for idx, weight_set in enumerate(weight_sets):
                print(f"\n\n{'=' * 60}")
                print(
                    f"اجرای مدل با {weight_set['name']} (w1={weight_set['w1']}, w2={weight_set['w2']}, w3={weight_set['w3']})")
                print(f"{'=' * 60}")

                # ایجاد فایل وزن موقت
                temp_weights_file = f"temp_weights_{idx}.json"
                with open(temp_weights_file, 'w', encoding='utf-8') as f:
                    json.dump({'w1': weight_set['w1'], 'w2': weight_set['w2'], 'w3': weight_set['w3']}, f,
                              ensure_ascii=False)

                # ایجاد و اجرای بهینه‌ساز با وزن‌های جدید
                temp_optimizer = VaccineAllocationOptimizer(excel_file, config_file, temp_weights_file)

                # اجرای مدل
                if find_optimal:
                    result = temp_optimizer.run_with_optimal_timing()
                else:
                    result = temp_optimizer.run(find_optimal_timing=False)

                if result:
                    # ذخیره نتایج با نام متفاوت
                    temp_optimizer.save_results_to_json(result, f"results_weightset_{idx + 1}.json")

                    # اضافه کردن به مجموعه نتایج
                    equity_metrics = result.get('equity_metrics', {})
                    results_collection.append({
                        'weight_set': weight_set,
                        'objective_value': result['objective_value'],
                        'objective1_value': result['objective1_value'],
                        'objective2_value': result['objective2_value'],
                        'objective3_value': result['objective3_value'],
                        'U1_1': result['U1'][1],
                        'U1_2': result['U1'][2],
                        'U2_1': result['U2'][1],
                        'U2_2': result['U2'][2],
                        'V_prime_1': result['V_prime'][1],
                        'V_prime_2': result['V_prime'][2],
                        'equity_diff_dose1': equity_metrics.get('equity_diff_dose1', 0),
                        'population_effectiveness': equity_metrics.get('population_effectiveness', 0)
                    })

                # حذف فایل موقت
                if os.path.exists(temp_weights_file):
                    os.remove(temp_weights_file)

            # نمایش مقایسه نتایج
            if results_collection:
                print("\n\n🎯 === مقایسه نتایج با وزن‌های مختلف (محدودیت‌های انعطاف‌پذیر) ===")
                header = "نام مجموعه وزن | هزینه کل | Z1 (تأمین) | Z2 (اجتماعی) | Z3 (اقتصادی) | U1_1 | U1_2 | U2_1 | U2_2 | V1 | V2 | عدالت | کارایی"
                print(header)
                print("-" * len(header))

                for result in results_collection:
                    print(
                        f"{result['weight_set']['name'][:20]:20} | {result['objective_value']:.2f} | {result['objective1_value']:.0f} | "
                        f"{result['objective2_value']:.0f} | {result['objective3_value']:.0f} | {result['U1_1']:.2f} | "
                        f"{result['U1_2']:.2f} | {result['U2_1']:.2f} | {result['U2_2']:.2f} | {result['V_prime_1']:.0f} | "
                        f"{result['V_prime_2']:.0f} | {result['equity_diff_dose1']:.3f} | {result['population_effectiveness']:.3f}")

                # ذخیره مقایسه در فایل
                with open("weight_comparison_results_flexible.json", 'w', encoding='utf-8') as f:
                    json.dump(results_collection, f, ensure_ascii=False, indent=4)

                print("\n✅ مقایسه نتایج در فایل 'weight_comparison_results_flexible.json' ذخیره شد.")

                # رسم نمودار مقایسه وزن‌ها
                try:
                    plt.figure(figsize=(14, 8))

                    weight_names = [ws['name'][:15] for ws in weight_sets]  # کوتاه کردن نام‌ها
                    u1_1_values = [r['U1_1'] for r in results_collection]
                    u1_2_values = [r['U1_2'] for r in results_collection]
                    equity_diff_values = [r['equity_diff_dose1'] for r in results_collection]

                    x = np.arange(len(weight_names))
                    width = 0.25

                    plt.bar(x - width, u1_1_values, width, label="نسبت واکسن گروه 1 (افراد بالای 60 سال)",
                            color='skyblue')
                    plt.bar(x, u1_2_values, width, label="نسبت واکسن گروه 2 (افراد دارای کسب و کار)",
                            color='lightgreen')
                    plt.bar(x + width, equity_diff_values, width, label="شاخص عدالت (اختلاف تخصیص)", color='orange')

                    plt.xlabel("مجموعه وزن‌ها")
                    plt.ylabel("مقدار")
                    plt.title("مقایسه تخصیص واکسن با محدودیت‌های انعطاف‌پذیر")
                    plt.xticks(x, weight_names, rotation=45)
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig("weight_comparison_flexible.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✅ نمودار مقایسه وزن‌ها ذخیره شد: weight_comparison_flexible.png")

                    # نمایش تفاوت‌های کلیدی
                    print("\n🎯 === تفاوت‌های کلیدی بین وزن‌ها ===")
                    min_u1_1 = min(r['U1_1'] for r in results_collection)
                    max_u1_1 = max(r['U1_1'] for r in results_collection)
                    min_u1_2 = min(r['U1_2'] for r in results_collection)
                    max_u1_2 = max(r['U1_2'] for r in results_collection)

                    print(
                        f"دامنه تغییرات گروه 1 (دوز اول): {min_u1_1:.2f} تا {max_u1_1:.2f} ({(max_u1_1 - min_u1_1) * 100:.1f}% تفاوت)")
                    print(
                        f"دامنه تغییرات گروه 2 (دوز اول): {min_u1_2:.2f} تا {max_u1_2:.2f} ({(max_u1_2 - min_u1_2) * 100:.1f}% تفاوت)")
                    print("✅ محدودیت‌های انعطاف‌پذیر به وزن‌ها اجازه تأثیرگذاری واقعی داده‌است!")

                except Exception as e:
                    print(f"خطا در رسم نمودار مقایسه وزن‌ها: {e}")
        else:
            # اجرای عادی مدل
            print("\n🎯 محدودیت‌های انعطاف‌پذیر فعال است!")
            optimizer.run(find_optimal_timing=find_optimal)

    except Exception as e:
        print(f"خطا در اجرای برنامه: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()