import tkinter as tk
from tkinter import messagebox, filedialog
import default_config as default_config  # Assuming this is your config file with the values provided above
import subprocess
import os
import sys
import threading
import queue


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
        canvas.create_window((0, 0), window=self.content_frame, anchor="ne")

        # Add the recording_config frame to the canvas
        self.recording_config_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.recording_config_frame, anchor="nw")
        
        self.stop_event = threading.Event()

        # File paths
        self.input_folder_label = tk.Label(self.content_frame, text="Input Folder Path: \n Contains .mat Files")
        self.input_folder_label.pack(pady=5)
        self.input_folder_entry = tk.Entry(self.content_frame, width=50)
        self.input_folder_entry.pack(pady=5)
        self.input_folder_entry.insert(0, default_config.input_folder_path)
        self.input_folder_button = tk.Button(self.content_frame, text="Browse", command=self.browse_input_folder)
        self.input_folder_button.pack(pady=5)

        def subtype_update(event):
            selected=self.scoring_subtypes_entry.curselection()
            text_value="Scoring Types Selected: \n"
            print(selected)
            output_text.set(text_value + str([ self.scoring_subtypes_entry.get(chan) for chan in selected ])) 

            if self.other_subtypes_entry.get() != [0,]:
                output_text.set(output_text.get()+str(self.other_subtypes_entry.get()))
            
            self.scoring_subtypes_output.update()
            

        
        self.scoring_subtypes_label = tk.Label(self.content_frame, text="Select Subtype Folders: \n Default: NoNo, SlNo, SlSz, NoNo")
        self.scoring_subtypes_label.pack(pady=5)
        listoptions=tk.StringVar(value=[*default_config.types,'Other'])
        self.scoring_subtypes_entry = tk.Listbox(self.content_frame, listvariable=listoptions, height=5, width=60,selectmode='extended',exportselection=0)
        self.scoring_subtypes_entry.bind('<<ListboxSelect>>', subtype_update)      
        self.scoring_subtypes_entry.pack(pady=5)

        self.other_subtypes_label = tk.Label(self.content_frame, text="Add Other Subtype Folders: \n As Comma-Separated List")
        self.other_subtypes_label.pack(pady=5)
        self.other_subtypes_entry = tk.Entry(self.content_frame, width=60)
        self.other_subtypes_entry.pack(pady=5)

        output_text=tk.StringVar(value='Scoring Types Selected:')
        self.scoring_subtypes_output = tk.Label(self.content_frame, textvar=output_text)
        self.scoring_subtypes_output.pack()

        self.export_folder_label = tk.Label(self.content_frame, text="Export Folder Path: \n will contain Decimated, Metadata, Py, \n and ScoredData folders")
        self.export_folder_label.pack(pady=5)
        self.export_folder_entry = tk.Entry(self.content_frame, width=50)
        self.export_folder_entry.pack(pady=5)
        self.export_folder_entry.insert(0, default_config.export_folder_path)
        self.export_folder_button = tk.Button(self.content_frame, text="Browse", command=self.browse_export_folder)
        self.export_folder_button.pack(pady=5)

        
        # Recording data parameters
        self.recording_length_label = tk.Label(self.recording_config_frame, text="Recording Length (hours):")
        self.recording_length_label.pack(pady=5)
        self.recording_length_entry = tk.Entry(self.recording_config_frame, width=10)
        self.recording_length_entry.pack(pady=5)
        self.recording_length_entry.insert(0, default_config.recording_length_hours)

        self.sampling_freq_label = tk.Label(self.recording_config_frame, text="Sampling Frequency (Hz):")
        self.sampling_freq_label.pack(pady=5)
        self.sampling_freq_entry = tk.Entry(self.recording_config_frame, width=10)
        self.sampling_freq_entry.pack(pady=5)
        self.sampling_freq_entry.insert(0, default_config.sampling_freq)

        self.epoch_length_label = tk.Label(self.recording_config_frame, text="Epoch Length (s):")
        self.epoch_length_label.pack(pady=5)
        self.epoch_length_entry = tk.Entry(self.recording_config_frame, width=10)
        self.epoch_length_entry.pack(pady=5)
        self.epoch_length_entry.insert(0, default_config.epoch_length)


        
        # # File paths
        # self.decimated_folder_label = tk.Label(self.content_frame, text="Decimated Folder Path:")
        # self.decimated_folder_label.pack(pady=5)
        # self.decimated_folder_entry = tk.Entry(self.content_frame, width=50)
        # self.decimated_folder_entry.pack(pady=5)
        # self.decimated_folder_entry.insert(0, default_config.decimated_folder_path)
        # self.decimated_folder_button = tk.Button(self.content_frame, text="Browse", command=self.browse_decimated_folder)
        # self.decimated_folder_button.pack(pady=5)

        # self.metadata_folder_label = tk.Label(self.content_frame, text="Metadata Folder Path:")
        # self.metadata_folder_label.pack(pady=5)
        # self.metadata_folder_entry = tk.Entry(self.content_frame, width=50)
        # self.metadata_folder_entry.pack(pady=5)
        # self.metadata_folder_entry.insert(0, default_config.metadata_folder_path)
        # self.metadata_folder_button = tk.Button(self.content_frame, text="Browse", command=self.browse_metadata_folder)
        # self.metadata_folder_button.pack(pady=5)

        # self.processed_data_folder_label = tk.Label(self.content_frame, text="Processed Data File Path:")
        # self.processed_data_folder_label.pack(pady=5)
        # self.processed_data_folder_entry = tk.Entry(self.content_frame, width=50)
        # self.processed_data_folder_entry.pack(pady=5)
        # self.processed_data_folder_entry.insert(0, default_config.processed_data_folder_path)
        # self.processed_data_folder_button = tk.Button(self.content_frame, text="Browse", command=self.browse_processed_file)
        # self.processed_data_folder_button.pack(pady=5)

        # Channels

        def update_input(event):
            selected=self.input_channels_entry.curselection()
            text_value="Selected Input Channels: \n"
            selected_input_text.set(text_value + str([ self.input_channels_entry.get(chan) for chan in selected ]) )
            print(selected_input_text.get())
            self.input_channels_selected_label.update()
            

            
        self.input_channels_label = tk.Label(self.recording_config_frame, text="Select Channels:")
        self.input_channels_label.pack(pady=5)
        self.input_channels_entry = tk.Listbox(self.recording_config_frame,height=5,width=60,exportselection=0)
        self.input_channels_entry.insert(tk.END,*default_config.input_channel_options)
        self.input_channels_entry.bind('<<ListboxSelect>>', update_input)

        self.input_channels_entry.pack(pady=5)

        selected_input_text=tk.StringVar(value="Selected Input Channels:")
        self.input_channels_selected_label = tk.Label(self.recording_config_frame, textvar=selected_input_text)
        self.input_channels_selected_label.pack()


        
        def update(event):
            selected=self.channels_entry.curselection()
            text_value="Selected Channels: \n"
            selected_text.set(text_value + str([ self.channels_entry.get(chan) for chan in selected ]) )
            print(selected_text.get())
            self.channels_selected_label.update()
            


        

        
        self.channels_label = tk.Label(self.recording_config_frame, text="Select Channels:")
        self.channels_label.pack(pady=5)
        self.channels_entry = tk.Listbox(self.recording_config_frame,height=5,width=60,exportselection=0)
        self.channels_entry.insert(tk.END,*default_config.channel_options)
        self.channels_entry.bind('<<ListboxSelect>>', update)

        self.channels_entry.pack(pady=5)



        
        selected_text=tk.StringVar(value="Selected Channels:")
        self.channels_selected_label = tk.Label(self.recording_config_frame, textvar=selected_text)
        self.channels_selected_label.pack()





        # self.channels_entry.insert(0, ", ".join(config.target_channels))

        # Feature Generation
        self.full_features_length_label = tk.Label(self.recording_config_frame, text="Input Features \n Default: 100 \n Allocate 20 Per Selected Non-EMG Channel \n 40 for EMG:")
        self.full_features_length_label.pack(pady=5)
        self.full_features_length_entry = tk.Entry(self.recording_config_frame, width=10)
        self.full_features_length_entry.pack(pady=5)
        self.full_features_length_entry.insert(0, default_config.full_features_length)

        # Feature Generation
        self.state_labels_entry_label = tk.Label(self.recording_config_frame, text="Input Score Markers \n Default: [Wake: 1, NREM: 2, REM: 3, Seizure: 4, Post-Ictal: 5] \n")
        self.state_labels_entry_label.pack(pady=5)

        self.wake_text = tk.Label(self.recording_config_frame, text='Wake: ')
        self.wake_text.pack()
        self.wake_int_entry = tk.Entry(self.recording_config_frame, width=1)
        self.wake_int_entry.pack(pady=0)
        self.wake_int_entry.insert(0, 1)

        self.nrem_text = tk.Label(self.recording_config_frame, text='NREM: ')
        self.nrem_text.pack()
        self.nrem_int_entry = tk.Entry(self.recording_config_frame, width=1)
        self.nrem_int_entry.pack(pady=0)
        self.nrem_int_entry.insert(0, 2)
        
        self.rem_text = tk.Label(self.recording_config_frame, text='REM: ')
        self.rem_text.pack()
        self.rem_int_entry = tk.Entry(self.recording_config_frame, width=1)
        self.rem_int_entry.pack(pady=0)
        self.rem_int_entry.insert(0, 3)
        
        self.seizure_text = tk.Label(self.recording_config_frame, text='Seizure: ')
        self.seizure_text.pack()
        self.seizure_int_entry = tk.Entry(self.recording_config_frame, width=1)
        self.seizure_int_entry.pack(pady=0)
        self.seizure_int_entry.insert(0, 4)
        
        self.postictal_text = tk.Label(self.recording_config_frame, text='Post-Ictal: ')
        self.postictal_text.pack()
        self.postictal_int_entry = tk.Entry(self.recording_config_frame, width=1)
        self.postictal_int_entry.pack(pady=0)
        self.postictal_int_entry.insert(0, 5)

        self.save_decimated_flag=tk.BooleanVar()
        self.save_decimated_button = tk.Checkbutton(self.recording_config_frame, text='Save Decimated .py Files',variable=self.save_decimated_flag,onvalue=True, offvalue=False)
        self.save_decimated_button.pack()

        
        # Save Button
        self.save_button = tk.Button(self.content_frame, text="Save Config", command=self.save_config)
        self.save_button.pack(pady=(20,0))

        # Run Button
        self.run_button = tk.Button(self.content_frame, text="Run Preprocessing Script", command=self.start_run_script)
        self.run_button.pack(pady=(20,0))

        # Run Button
        self.run_button = tk.Button(self.content_frame, text="Run Postprocessing Script", command=self.start_run_post_script)
        self.run_button.pack(pady=(20,0))

        self.console_text = tk.Text(self.content_frame, state='disabled', width=60, height=5,yscrollcommand=True)
        self.console_text.pack(pady=(20,0))

        # We redirect sys.stdout -> TextRedirector
        self.log_queue = queue.Queue()
        sys.stdout = TextRedirector(self.log_queue)
        sys.stderr = TextRedirector(self.log_queue)
        
        self.root.after(50, self.poll_log_queue)

    def background(task):
        """Creates and starts a new thread for the long_running_task."""
        # Create a new thread targeting the long_running_task function
        thread = threading.Thread(target=task)
        # Start the thread
        thread.start()
         
    def browse_input_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.input_folder_entry.delete(0, tk.END)
            self.input_folder_entry.insert(0, folder_selected)
            
    def browse_export_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.export_folder_entry.delete(0, tk.END)
            self.export_folder_entry.insert(0, folder_selected)

    def save_config(self):
        try:
            # Retrieve values from the entries and update config.py

            new_input_folder_path=str(self.input_folder_entry.get())

            selection = self.scoring_subtypes_entry.curselection()
            scoring_subtypes_list=[f'{self.scoring_subtypes_entry.get(folder)}' for folder in selection ]
            scoring_subtypes_list=[folder for folder in scoring_subtypes_list if folder not in 'Other']
            
            other_subtypes=[f'/{folder}/' for folder in list(self.other_subtypes_entry.get())]
            new_types= scoring_subtypes_list + other_subtypes
            
            new_recording_length_hours = int(self.recording_length_entry.get())
            new_sampling_freq = int(self.sampling_freq_entry.get())
            new_epoch_length = int(self.epoch_length_entry.get())
            new_export_folder = str(self.export_folder_entry.get())

            selection = self.input_channels_entry.curselection()
            new_input_channels = [ch.strip() for ch in self.input_channels_entry.get( selection )]
            
            selection = self.channels_entry.curselection()
            new_channels = [ch.strip() for ch in self.channels_entry.get( selection )]
            
            new_full_features_length = int(self.full_features_length_entry.get())

            input_scores=[int(self.wake_int_entry.get()),
                          int(self.nrem_int_entry.get()),
                          int(self.rem_int_entry.get()),
                          int(self.seizure_int_entry.get()),
                          int(self.postictal_int_entry.get())]

            # Update config.py file
            with open('config.py', 'w') as f:
                f.write(f"import scipy.signal as signal \n")
                f.write(f"recording_length_hours = {new_recording_length_hours}\n")
                f.write(f"sampling_freq = {new_sampling_freq}\n")
                f.write(f"epoch_length = {new_epoch_length}\n")

                f.write(f"recording_length_seconds = {int(new_recording_length_hours*3600)}\n")
                f.write(f"target_epoch_count= {int(new_recording_length_hours*3600/new_epoch_length)}\n")
                f.write(f"sampling_freq_dec = {int(200)}\n")
                f.write(f"epoch_samples_dec = {int(200*new_epoch_length)}\n")
                f.write(f"resampling_factor={int(new_sampling_freq/200)}\n")

                f.write(f"input_channels = {new_input_channels}\n")

                
                f.write(f"target_channels = {new_channels}\n")
                f.write(f"states = {default_config.states}\n")
                f.write(f"input_scores = {input_scores}\n")
                f.write(f"target_scores = {default_config.target_scores}\n")

                f.write(f"save_decimated = {self.save_decimated_flag.get()}\n")
                
                f.write(f'input_path = "{new_input_folder_path}/"')
                f.write("\n")

                f.write(f'types = {new_types} \n')

                f.write(f'export_folder_path = "{new_export_folder}"')
                f.write("\n")
                f.write(f'decimated_folder_path = "{new_export_folder}/Decimated"')
                f.write("\n")
                f.write(f'metadata_folder_path = "{new_export_folder}/Metadata"')
                f.write("\n")

                f.write(f'processed_data_folder_path = "{new_export_folder}/Py"')
                f.write("\n")

                f.write(f'scoring_output_path = "{new_export_folder}/ScoredData"')
                f.write("\n")

                f.write(f'log_file_path = "{new_export_folder}/export_log.txt"')
                f.write("\n")

                f.write('folders=[export_folder_path,decimated_folder_path,metadata_folder_path,processed_data_folder_path,scoring_output_path]')
                f.write("\n")

                
                f.write(f"b,a = signal.butter(1,[1],'high', fs=sampling_freq)\n")
                f.write("\n")

                f.write(f"full_features_length = {new_full_features_length}\n")

                #recording_length_seconds=int(recording_length_hours*3600)
                #target_epoch_count=int(recording_length_seconds/epoch_length)
                # sampling_freq_dec=int(sampling_freq/10)    
                # epoch_samples_dec=sampling_freq_dec*epoch_length 


            if os.path.exists(new_export_folder)==False:
                print(f'creating {new_export_folder}') 
                os.mkdir(new_export_folder)
            for folders in ['/Decimated','/Metadata','/Py','/ScoredData']:
                if os.path.exists(new_export_folder+folders)==False:
                    os.mkdir(new_export_folder+folders)

            messagebox.showinfo("Config Saved", "Configuration saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


    def poll_log_queue(self):
        try:
            while True:
                text = self.log_queue.get_nowait()
                self.console_text.configure(state='normal')
                self.console_text.insert(tk.END, text)
                self.console_text.configure(state='disabled')
                self.console_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(50, self.poll_log_queue)
    
    def stop_script(self):
        self.stop_event.set()
    
    def run_script(self):
        try:
            # Run the process_file.py script using the updated config.py
            import config
            import preprocessing
            preprocessing.process_and_save(config.input_path)
            messagebox.showinfo("Script Run", "Script ran successfully!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"An error occurred while running the script: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def run_post_script(self):
        try:
            # Run the process_file.py script using the updated config.py
            import postprocessing
            postprocessing.model_predict(config.processed_data_folder_path)
            messagebox.showinfo("Script Run", "Script ran successfully!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"An error occurred while running the script: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def start_run_script(self):
        self.worker_thread = threading.Thread(
            target=self.run_script,
            daemon=True
        )
        self.worker_thread.start()

class TextRedirector:
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigEditorApp(root)
    root.geometry("1000x1000")  # Set the window size
    root.update_idletasks()

    root.mainloop()
