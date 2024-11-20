### **Short Instructions for Working with the Project**

#### **First Time Setup**
1. **Create and Activate the Environment**:
   - Create the Conda environment using the `environment.yml` file:
     ```bash
     conda env create -f environment.yml
     ```
   - Activate the environment:
     ```bash
     conda activate dz6
     ```

2. **Register the Environment with Jupyter (Optional, if using Jupyter)**:
   - Register the `dz6` environment as a Jupyter kernel:
     ```bash
     python -m ipykernel install --user --name dz6 --display-name "Python (dz6)"
     ```

3. **Verify the Setup**:
   - Test imports to ensure all packages are correctly installed:
     ```bash
     python -c "import fastai; import pandas; print('Setup complete!')"
     ```

---

#### **Starting Work (After the First Time)**
1. **Activate the Environment**:
   ```bash
   conda activate dz6
   ```

2. **Launch Jupyter Notebook (If Using Jupyter)**:
   ```bash
   jupyter notebook
   ```
   - Select the kernel **Python (dz6)** if working with notebooks.

3. **Run Your Code**:
   - Open your script or notebook and start working!

---