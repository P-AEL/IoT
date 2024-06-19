# IoT
Displaying Room features.
Doing EDA.
Predicting future values with Transformer/LSTM

* Installation
* Starting the Application

![Screenshot](readme_source/Screenshot 2024-06-19 142528.png)


Clone repository:

git clone https://github.com/P-AEL/IoT.git


Install dependency:

pip install -r requirements.txt

Starting the dashboard:

streamlit run ./main/dashboard.py

Known Problems:

* The pyvista roomplot doesnt work on mac
* The describe function in the eda page creates a warning (is dependend on the pyarrow version) but doesnt afflict the functionality
* Depending on your streamlit version you might need to delete the parameters in this line: "st.container(border=True):"