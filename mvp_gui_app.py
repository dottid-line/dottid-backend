# mvp_gui_app.py

import os
import shutil
import datetime
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

from orchestrator import run_full_underwrite
from inference_engine import classify_images

BASE_DIR = r"C:\FINAL MVP MODEL"
EXPORT_ROOT = os.path.join(BASE_DIR, "gui_runs")
os.makedirs(EXPORT_ROOT, exist_ok=True)


class UnderwritingGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Dottid Line – MVP Underwriting System")
        self.geometry("1600x950")

        self.image_paths = []
        self.classified_images = []
        self.unified_result = None

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.tab_images = ttk.Frame(self.notebook)
        self.tab_info = ttk.Frame(self.notebook)
        self.tab_processing = ttk.Frame(self.notebook)
        self.tab_rehab = ttk.Frame(self.notebook)
        self.tab_export = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_images, text="Images")
        self.notebook.add(self.tab_info, text="Property Info")
        self.notebook.add(self.tab_processing, text="Processing Status")
        self.notebook.add(self.tab_rehab, text="Underwriting Results")
        self.notebook.add(self.tab_export, text="Export")

        self.build_images_tab()
        self.build_info_tab()
        self.build_processing_tab()
        self.build_rehab_tab()
        self.build_export_tab()


    ###########################################################################
    # PROCESSING STATUS TAB
    ###########################################################################
    def build_processing_tab(self):
        f = self.tab_processing

        self.log_box = tk.Text(f, width=150, height=45, state="disabled")
        self.log_box.pack(padx=20, pady=20, fill="both", expand=True)

        self.processing_status = tk.Label(f, text="", font=("Arial", 14), fg="blue")
        self.processing_status.pack(pady=5)

        self.notebook.tab(2, state="disabled")

    def log(self, message):
        self.log_box.config(state="normal")
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state="disabled")
        self.update_idletasks()


    ###########################################################################
    # IMAGES TAB
    ###########################################################################
    def build_images_tab(self):
        top_bar = tk.Frame(self.tab_images)
        top_bar.pack(fill="x", pady=10)

        tk.Button(top_bar, text="Upload Images", command=self.upload_images).pack(side="left", padx=10)
        tk.Button(top_bar, text="Analyze Images", command=self.analyze_images).pack(side="left", padx=10)
        tk.Button(top_bar, text="Next → Property Info", command=self.goto_property_info).pack(side="left", padx=10)
        tk.Button(top_bar, text="Create New Run", command=self.create_new_run).pack(side="left", padx=10)

        container = tk.Frame(self.tab_images)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.images_inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.images_inner, anchor="nw")

        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.images_inner.bind("<Configure>", on_configure)
        self.canvas = canvas

    def create_new_run(self):
        self.image_paths = []
        self.classified_images = []
        self.unified_result = None

        for w in self.images_inner.winfo_children():
            w.destroy()
        self.rehab_output.delete("1.0", tk.END)

        if hasattr(self, "export_status"):
            self.export_status.config(text="")

        if hasattr(self, "assignment_entry"):
            self.assignment_entry.delete(0, tk.END)

        self.notebook.tab(2, state="disabled")
        self.notebook.select(self.tab_images)

    def upload_images(self):
        paths = filedialog.askopenfilenames(
            title="Select Property Images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not paths:
            return
        self.image_paths.extend(paths)
        self.render_images()

    def analyze_images(self):
        if not self.image_paths:
            return
        self.log("Analyzing images...")
        self.classified_images = classify_images(self.image_paths)
        self.render_images()
        self.log("Image analysis complete.")

    def render_images(self):
        for w in self.images_inner.winfo_children():
            w.destroy()

        for p in self.image_paths:
            row = tk.Frame(self.images_inner, pady=5)
            row.pack(fill="x", anchor="w")

            try:
                im = Image.open(p).convert("RGB")
                thumb = im.resize((120, 120))
                tk_img = ImageTk.PhotoImage(thumb)
                img_lbl = tk.Label(row, image=tk_img)
                img_lbl.image = tk_img
                img_lbl.pack(side="left", padx=10)
            except:
                tk.Label(row, text="[image load error]").pack(side="left", padx=10)

            text_frame = tk.Frame(row)
            text_frame.pack(side="left", padx=10)

            tk.Label(text_frame, text=os.path.basename(p)).pack(anchor="w")

            match = next((x for x in self.classified_images if x["image_path"] == p), None)
            if match:
                tk.Label(text_frame, text=f"VALID: {match['valid']} ({match['valid_conf']:.2f})").pack(anchor="w")
                tk.Label(text_frame, text=f"ROOM: {match['room_type']} ({match['room_conf']:.2f})").pack(anchor="w")
                tk.Label(text_frame, text=f"COND: {match['condition']} ({match['condition_conf']:.2f})").pack(anchor="w")

    def goto_property_info(self):
        self.notebook.select(self.tab_info)


    ###########################################################################
    # PROPERTY INFO TAB
    ###########################################################################
    def build_info_tab(self):
        f = self.tab_info

        labels = [
            "property_address",
            "property_type",
            "beds",
            "baths",
            "sqft",
            "year_built",
            "number_of_units",
            "kitchen_last_updated (years)",
            "bath_last_updated (years)",
            "roof_replacement_needed",
            "hvac_replacement_needed",
            "foundation_issues",
            "buyer_type (rental / flip / wholesale)",
            "assignment_fee_if_wholesale"
        ]

        for i, text in enumerate(labels):
            tk.Label(f, text=text).grid(row=i, column=0, sticky="w", padx=10, pady=5)

        self.entry_address = tk.Entry(f, width=50); self.entry_address.grid(row=0, column=1)

        self.cb_property_type = ttk.Combobox(
            f,
            values=["single_family", "multi_family", "condo", "townhouse"],
            state="readonly"
        )
        self.cb_property_type.grid(row=1, column=1)

        self.entry_beds = tk.Entry(f); self.entry_beds.grid(row=2, column=1)
        self.entry_baths = tk.Entry(f); self.entry_baths.grid(row=3, column=1)
        self.entry_sqft = tk.Entry(f); self.entry_sqft.grid(row=4, column=1)
        self.entry_year_built = tk.Entry(f); self.entry_year_built.grid(row=5, column=1)

        self.entry_units = tk.Entry(f); self.entry_units.grid(row=6, column=1)

        self.entry_kitchen_age = tk.Entry(f); self.entry_kitchen_age.grid(row=7, column=1)
        self.entry_bath_age = tk.Entry(f); self.entry_bath_age.grid(row=8, column=1)

        self.cb_roof_needed = ttk.Combobox(f, values=["no", "yes"], state="readonly")
        self.cb_roof_needed.grid(row=9, column=1)

        self.cb_hvac_needed = ttk.Combobox(f, values=["no", "yes"], state="readonly")
        self.cb_hvac_needed.grid(row=10, column=1)

        self.cb_foundation = ttk.Combobox(
            f,
            values=["none", "minor_issues", "major_issues"],
            state="readonly"
        )
        self.cb_foundation.grid(row=11, column=1)

        self.cb_buyer_type = ttk.Combobox(
            f,
            values=["rental", "flip", "wholesale"],
            state="readonly"
        )
        self.cb_buyer_type.grid(row=12, column=1)
        self.cb_buyer_type.bind("<<ComboboxSelected>>", self.on_buyer_type_change)

        self.assignment_label = tk.Label(f, text="Assignment Fee ($)")
        self.assignment_entry = tk.Entry(f, width=20)

        self.assignment_label.grid(row=13, column=0, sticky="w", padx=10, pady=5)
        self.assignment_entry.grid(row=13, column=1, sticky="w", padx=10, pady=5)

        self.assignment_label.grid_remove()
        self.assignment_entry.grid_remove()

        tk.Button(f, text="Submit Property Info → Run Underwriting", command=self.launch_processing).grid(row=14, column=1, pady=10)


    ###########################################################################
    # WHOLESALE FIELD BEHAVIOR
    ###########################################################################
    def on_buyer_type_change(self, event=None):
        buyer_type = (self.cb_buyer_type.get() or "").strip().lower()

        if buyer_type == "wholesale":
            self.assignment_label.grid()
            self.assignment_entry.grid()
        else:
            self.assignment_label.grid_remove()
            self.assignment_entry.grid_remove()


    ###########################################################################
    # PROCESSING ENGINE
    ###########################################################################
    def log_stage(self, msg):
        self.log(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

    def launch_processing(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.config(state="disabled")

        self.notebook.tab(2, state="normal")
        self.notebook.select(self.tab_processing)

        t = threading.Thread(target=self.compute_underwriting)
        t.start()

    def compute_underwriting(self):
        try:
            self.log_stage("VALIDATING INPUT...")

            addr = self.entry_address.get().strip()
            prop_type = self.cb_property_type.get().strip().lower()
            beds = float(self.entry_beds.get().strip() or 0)
            baths = float(self.entry_baths.get().strip() or 0)
            sqft = float(self.entry_sqft.get().strip() or 0)
            year_built = int(self.entry_year_built.get().strip() or 0)
            units = int(self.entry_units.get().strip() or 1)
            kitchen_age = float(self.entry_kitchen_age.get().strip() or 50)
            bath_age = float(self.entry_bath_age.get().strip() or 50)
            roof_needed = (self.cb_roof_needed.get() == "yes")
            hvac_needed = (self.cb_hvac_needed.get() == "yes")
            foundation_issues = (self.cb_foundation.get() or "none")
            buyer_type = (self.cb_buyer_type.get() or "").strip().lower()

            assignment_fee = 0.0
            if buyer_type == "wholesale":
                try:
                    assignment_fee = float(self.assignment_entry.get().strip() or 0.0)
                except:
                    assignment_fee = 0.0

            subject = {
                "address": addr,
                "property_type": prop_type,
                "beds": beds,
                "baths": baths,
                "sqft": sqft,
                "year_built": year_built,
                "units": units,
                "kitchen_age": kitchen_age,
                "bath_age": bath_age,
                "roof_needed": roof_needed,
                "hvac_needed": hvac_needed,
                "foundation_issues": foundation_issues,
                "assignment_fee": assignment_fee,
                "deal_type": buyer_type,
                "subject_images": self.image_paths,
            }

            self.log_stage("RUNNING FULL UNDERWRITE PIPELINE...")
            self.unified_result = run_full_underwrite(subject, logger=self.log_stage)

            self.log_stage("PROCESSING COMPLETE.")
            self.update_results_screen()
            self.notebook.select(self.tab_rehab)

        except Exception as e:
            self.log_stage(f"ERROR: {e}")


    ###########################################################################
    # RESULTS TAB
    ###########################################################################
    def build_rehab_tab(self):
        f = self.tab_rehab

        self.rehab_output = tk.Text(f, width=160, height=50)
        self.rehab_output.pack(padx=10, pady=10)

    def update_results_screen(self):
        r = self.unified_result
        rehab = r["rehab"]
        arv = r["arv"]
        mao = r["mao"]
        buyer_type = r["subject"]["deal_type"]

        self.rehab_output.delete("1.0", tk.END)

        arv_display = "<<NO ARV>>"
        try:
            arv_raw = arv.get("arv", {})
            arv_val = arv_raw.get("arv", None)
            if isinstance(arv_val, (int, float)):
                arv_display = f"${arv_val:,.0f}"
        except:
            pass

        self.rehab_output.insert(tk.END, f"BUYER TYPE: {buyer_type.upper() if buyer_type else 'N/A'}\n\n")
        self.rehab_output.insert(tk.END, f"ARV: {arv_display}\n\n")
        self.rehab_output.insert(tk.END, f"ESTIMATED REHAB: {rehab.get('estimate_str')}\n")
        self.rehab_output.insert(tk.END, f"MAX ALLOWABLE OFFER (MAO): {mao.get('mao_formatted')}\n\n")

        try:
            comps = arv.get("arv", {}).get("selected_comps", [])
            if comps:
                self.rehab_output.insert(tk.END, "---------------------------\n")
                self.rehab_output.insert(tk.END, "COMPARABLE SALES USED:\n\n")

                for i, c in enumerate(comps, start=1):
                    sold = c.get("sold_price")
                    beds = c.get("comp_beds") or c.get("beds")
                    baths = c.get("comp_baths") or c.get("baths")
                    sqft = c.get("comp_sqft")
                    dist = c.get("distance_miles")
                    thumb = c.get("image_url")
                    link = c.get("detail_url")

                    self.rehab_output.insert(tk.END, f"Comp {i}:\n")
                    self.rehab_output.insert(tk.END, f"  Address: {c.get('address')}\n")
                    self.rehab_output.insert(tk.END, f"  Sold Price: ${sold:,.0f}\n")
                    self.rehab_output.insert(tk.END, f"  Beds/Baths: {beds} / {baths}\n")
                    self.rehab_output.insert(tk.END, f"  Sqft: {sqft:,}\n")
                    self.rehab_output.insert(tk.END, f"  Distance: {dist:.2f} mi\n")
                    self.rehab_output.insert(tk.END, f"  Zillow Link: {link}\n\n")

        except:
            pass


    ###########################################################################
    # EXPORT TAB
    ###########################################################################
    def build_export_tab(self):
        f = self.tab_export

        tk.Button(f, text="Export This Run", command=self.export_run).pack(pady=20)
        self.export_status = tk.Label(f, text="", wraplength=1000)
        self.export_status.pack()

    def export_run(self):
        addr_raw = self.entry_address.get().strip()
        if not addr_raw:
            self.export_status.config(text="ERROR: Address is required.")
            return

        addr = ", ".join(part.strip() for part in addr_raw.split(","))

        today = datetime.date.today().strftime("%Y-%m-%d")
        folder_name = f"{addr} ({today})"
        export_dir = os.path.join(EXPORT_ROOT, folder_name)

        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        os.makedirs(export_dir, exist_ok=True)

        report_path = os.path.join(export_dir, "Underwriting_Report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.rehab_output.get("1.0", tk.END))

        img_folder = os.path.join(export_dir, "images")
        os.makedirs(img_folder, exist_ok=True)

        for src in self.image_paths:
            dst = os.path.join(img_folder, os.path.basename(src))
            try:
                shutil.copy2(src, dst)
            except:
                pass

        self.export_status.config(text=f"Exported to: {export_dir}")


if __name__ == "__main__":
    app = UnderwritingGUI()
    app.mainloop()
