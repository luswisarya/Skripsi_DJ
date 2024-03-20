import tkinter as tk
from tkinter import ttk  # Import for Treeview
from connection import connector
# Assuming you have database.py with the functionality from previous discussions

class MyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("My Table Application")

        # Data retrieval (replace with actual call to your database function)
        self.data = connector()  # Assuming database.py has this function

        self.search_by = tk.StringVar()
        self.search_by.set("Nama Mahasiswa")

        style = ttk.Style()
        style.configure("Treeview.Heading",background="black",foreground="red", relief="raised")

        # Create Treeview
        self.tree = ttk.Treeview(self, columns=("#1", "#2", "#3", "#4","#5","#6","#7","#8","#9", "#10"), show="headings")  # Adjust columns as needed
        self.tree.heading("#1", text="No", anchor="center") 
        self.tree.column("#1", width=50, anchor="center") 

        self.tree.heading("#2", text="Nama Mahasiswa")
        self.tree.column("#2", width=250) 

        self.tree.heading("#3", text="Waktu Mulai")
        self.tree.column("#3", width=200)  

        self.tree.heading("#4", text="Total Frame")  
        self.tree.column("#4", width=100, anchor="center")

        self.tree.heading("#5", text="Ada Muka")  
        self.tree.column("#5", width=100, anchor="center")

        self.tree.heading("#6", text="Tidak Ada Muka / Berbeda") 
        self.tree.column("#6", width=150, anchor="center")

        self.tree.heading("#7", text="Jumlah Warning")  
        self.tree.column("#7", width=100, anchor="center")

        self.tree.heading("#8", text="Waktu Selesai")  
        self.tree.column("#8", width=250)

        self.tree.heading("#9", text="Durasi") 
        self.tree.column("#9", width=50, anchor="center") 

        self.tree.heading("#10", text="Level") 
        self.tree.column("#10", width=50, anchor="center") 

        # Insert data into Treeview
        for row in self.data:
            self.tree.insert("", tk.END, values=row)

        # Search widgets
        search_label = ttk.Label(self, text="Cari Berdasarkan:")
        search_combo = ttk.Combobox(self, values=["Nama Mahasiswa", "Waktu Mulai"], textvariable=self.search_by)
        search_entry = ttk.Entry(self)
        search_button = ttk.Button(self, text="Cari", command=lambda: self.search(self.search_by.get(), search_entry.get()))


        # Layout
        spacer = ttk.Label(self, width=0)
        search_label.grid(row=0, column=0, sticky="W", padx=10)
        search_combo.grid(row=0, column=1, sticky="W", pady=10)
        search_entry.grid(row=0, column=2, sticky="W", pady=10)
        search_button.grid(row=0, column=3, sticky="W", pady=10)
        self.tree.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=10, pady=10)

        reset_button = ttk.Button(self, text="Refresh")
        reset_button.grid(row=2, column=3, sticky="W", pady=10)
        reset_button.config(command=self.load_initial_data)


    def load_initial_data(self):
        self.data = connector()
        self.tree.delete(*self.tree.get_children())
        for row in self.data:
            self.tree.insert("", tk.END, values=row)


    def search(self, criteria, value):
        # Clear existing data
        self.tree.delete(*self.tree.get_children())

        # Loop through data and search based on criteria
        for row in self.data:
            if criteria == "Nama Mahasiswa" and value.lower() in row[1].lower():
                self.tree.insert("", tk.END, values=row)
            elif criteria == "Waktu Mulai" and value.lower() in row[2].lower():
                self.tree.insert("", tk.END, values=row)


# Run the GUI application
if __name__ == "__main__":
    gui = MyGUI()
    gui.mainloop()
