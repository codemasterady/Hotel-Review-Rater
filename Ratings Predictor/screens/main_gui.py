# Importing the libraries
from tkinter import *


#! Class
class GraphicalUserInterface():
    def __init__(self):
        self.root = Tk()
        self.root.title("Review")

    #! Generate The Main GUI
    def generateMainGUI(self):
        # Setting the GUI
        bck_canvas = Canvas(master=self.root, height=500,
                            width=500, bg='purple')
        bck_canvas.pack()
        main_label = Label(bck_canvas, text="Review", fg='white',
                           bg='#ba03fc', font=("Courier", 44))
        main_label.place(relx=0.2, rely=0.1, relwidth=0.6, relheight=0.2)
        review_text_field = Text(bck_canvas)
        review_text_field.place(
            relx=0.2, rely=0.4, relwidth=0.6, relheight=0.4)
        generate_btn = Button(bck_canvas, text="Generate",
                              bg='#7303fc', fg='white')
        generate_btn.place(relx=0.2, rely=0.85, relwidth=0.6, relheight=0.1)
        self.root.mainloop()

    #! Generate The Side GUI
    def generateReviewDisplay(self):
        main_canvas = Canvas(master=self.side_root, height=300,
                             width=300, bg='purple')
        main_canvas.pack()
        main_label = Label(main_canvas, text="Review", fg='white',
                           bg='#ba03fc', font=("Courier", 44))
        main_label.place(relx=0.2, rely=0.1, relwidth=0.6, relheight=0.2)

    #! Navigate to the review screen
    def __generateReview(self, num_of_stars):
        pass
