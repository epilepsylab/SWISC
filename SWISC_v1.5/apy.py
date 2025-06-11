import tkinter as tk
from tkinter import messagebox, filedialog
import config  # Assuming this is your config file with the values provided above
import subprocess
import os

class ConfigEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Config Editor")

        # Create a main frame inside the canvas
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=1)

        # Create a canvas widget
        canvas = tk.Canvas(main_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Add a scrollbar to the canvas
        scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas to work with the scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Create a second frame inside the canvas for content
        self.content_frame = tk.Frame(canvas)

        # Add the content frame to the canvas
        canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        # Now, add all the widgets to the `self.content_frame`

        # Recording data parameters
        self.recording_length_label = tk.Label(self.content_frame, text="Recording Length (hours):")
        self.recording_length_label.pack(pady=5)
        self.recording_length_entry = tk.Entry(self.content_frame, width=10)
        self.recording_length_entry.pack(pady=5)
        self.recording_length_entry.insert(0, config.recording_length_hours)

        self.sampling_freq_label = tk.Label(self.content_frame, text="Sampling Frequency (Hz):")
        self.sampling_freq_label.pack(pady=5)
        self.sampling_freq_entry = tk.Entry(self.content_frame, width=10)
        self.sampling_freq_entry.pack(pady=5)
        self.sampling_freq_entry.insert(0, config.sampling_freq)

        self.epoch_length_label = tk.Label(self.content_frame, text="Epoch Length (s):")
        self.epoch_length_label.pack(pady=5)
        self.epoch_length_entry = tk.Entry(self.content_frame, width=10)
        self.epoch_length_entry.pack(pady=5)
        self.epoch_length_entry.insert(0, config.epoch_length)

        # File paths
        self.decimated_folder_label = tk.Label(self.content_frame, text="Decimated Folder Path:")
        self.decimated_folder_label.pack(pady=5)
        self.decimated_folder_entry = tk.Entry(self.content_frame, width=50)
        self.decimated_folder_entry.pack(pady=5)
        self.decimated_folder_entry.insert(0, config.decimated_folder_path)
        self.decimated_folder_button = tk.Button(self.content_frame, text="Browse", command=self.browse_decimated_folder)
        self.decimated_folder_button.pack(pady=5)

        self.metadata_folder_label = tk.Label(self.content_frame, text="Metadata Folder Path:")
        self.metadata_folder_label.pack(pady=5)
        self.metadata_folder_entry = tk.Entry(self.content_frame, width=50)
        self.metadata_folder_entry.pack(pady=5)
        self.metadata_folder_entry.insert(0, config.metadata_folder_path)
        self.metadata_folder_button = tk.Button(self.content_frame, text="Browse", command=self.browse_metadata_folder)
        self.metadata_folder_button.pack(pady=5)

        self.processed_data_folder_label = tk.Label(self.content_frame, text="Processed Data File Path:")
        self.processed_data_folder_label.pack(pady=5)
        self.processed_data_folder_entry = tk.Entry(self.content_frame, width=50)
        self.processed_data_folder_entry.pack(pady=5)
        self.processed_data_folder_entry.insert(0, config.processed_data_folder_path)
        self.processed_data_folder_button = tk.Button(self.content_frame, text="Browse", command=self.browse_processed_file)
        self.processed_data_folder_button.pack(pady=5)

        # Channels
        self.channels_label = tk.Label(self.content_frame, text="Channels (comma-separated):")
        self.channels_label.pack(pady=5)
        self.channels_entry = tk.Entry(self.content_frame, width=50)
        self.channels_entry.pack(pady=5)
        self.channels_entry.insert(0, ", ".join(config.channels))

        # Feature Generation
        self.full_features_length_label = tk.Label(self.content_frame, text="Full Features Length:")
        self.full_features_length_label.pack(pady=5)
        self.full_features_length_entry = tk.Entry(self.content_frame, width=10)
        self.full_features_length_entry.pack(pady=5)
        self.full_features_length_entry.insert(0, config.full_features_length)

        # Save Button
        self.save_button = tk.Button(self.content_frame, text="Save Config", command=self.save_config)
        self.save_button.pack(pady=10)

        # Run Button
        self.run_button = tk.Button(self.content_frame, text="Run Script", command=self.run_script)
        self.run_button.pack(pady=10)

    def browse_decimated_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.decimated_folder_entry.delete(0, tk.END)
            self.decimated_folder_entry.insert(0, folder_selected)

    def browse_metadata_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.metadata_folder_entry.delete(0, tk.END)
            self.metadata_folder_entry.insert(0, folder_selected)

    def browse_processed_file(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.metadata_folder_entry.delete(0, tk.END)
            self.metadata_folder_entry.insert(0, folder_selected)

    def save_config(self):
        try:
            # Retrieve values from the entries and update config.py
            new_recording_length_hours = int(self.recording_length_entry.get())
            new_sampling_freq = int(self.sampling_freq_entry.get())
            new_epoch_length = int(self.epoch_length_entry.get())
            new_decimated_folder = self.decimated_folder_entry.get()
            new_processed_data_folder = self.processed_data_folder_entry.get()
            new_channels = [ch.strip() for ch in self.channels_entry.get().split(',')]
            new_full_features_length = int(self.full_features_length_entry.get())

            # Update config.py file
            with open('config.py', 'w') as f:
                f.write(f"recording_length_hours = {new_recording_length_hours}\n")
                f.write(f"sampling_freq = {new_sampling_freq}\n")
                f.write(f"epoch_length = {new_epoch_length}\n")

                f.write(f"recording_length_seconds = {int(new_recording_length_hours*3600)}\n")
                f.write(f"target_epoch_count= {int(new_recording_length_hours*3600/new_epoch_length)}\n")
                f.write(f"sampling_freq_dec = {int(new_sampling_freq/10)}\n")
                f.write(f"epoch_samples_dec = '{int(new_sampling_freq*new_epoch_length/10)}'\n")

             
                f.write(f"processed_data_folder_path = '{new_processed_data_folder}'\n")
                f.write(f"channels = {new_channels}\n")
                f.write(f"full_features_length = {new_full_features_length}\n")

                #recording_length_seconds=int(recording_length_hours*3600)
#target_epoch_count=int(recording_length_seconds/epoch_length)
# sampling_freq_dec=int(sampling_freq/10)    
# epoch_samples_dec=sampling_freq_dec*epoch_length 


                
              

            messagebox.showinfo("Config Saved", "Configuration saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def run_script(self):
        try:
            # Run the process_file.py script using the updated config.py
            script_path = os.path.join(os.getcwd(), 'postprocessing.py')
            subprocess.run(['python', script_path], check=True)
            messagebox.showinfo("Script Run", "Script ran successfully!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"An error occurred while running the script: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigEditorApp(root)
    root.geometry("600x600")  # Set the window size
    root.mainloop()
