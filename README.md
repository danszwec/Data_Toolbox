# Data_Toolbox

A collection of utility scripts and configurations designed to streamline data processing workflows.

## 🛠️ Features

- **optical_flow.py**: Implements optical flow algorithms for motion tracking.
- **utils.py**: Provides various utility functions to assist with data manipulation and processing tasks.
- **cfg.yaml**: Configuration file to manage settings and parameters for the scripts.

## 🚀 Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/danszwec/Data_Toolbox.git
cd Data_Toolbox
```


Install dependencies (if any):

```bash
pip install -r requirements.txt
```



## ⚙️ `cfg.yaml`

The `cfg.yaml` file contains configurable parameters for the scripts. Edit this file to adjust how optical flow analysis is performed.

- **`classify_by`**: Determines how the script groups the input data.  
  - `"folder"` → Treats each folder as a separate sequence for optical flow analysis.  
  - `"frame"` → Treats each frame individually.  

- **`motion_threshold`**: A numeric threshold to filter or detect motion. Only motion above this value will be considered significant.  

- **`input_path`**: The path to the directory containing input images or videos.  

- **`folders`**: A list of folder names inside the `input_path` that the script will process. Only these folders will be analyzed if `classify_by` is `"folder"`.



## 📄 Usage
optical_flow.py

Run optical flow processing on a video or image sequence:

```bash
python optical_flow.py --input <input_file> --output <output_file>
```




