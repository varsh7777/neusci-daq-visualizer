
# neusci-daq-visualizer

Application designed to visualize recordings from data acquisition systems for students within the University of Washington's undergraduate neuroscience program. Designed for and tested with NI USB-6003 from National Instruments using PyDAQmx with the NIDAQmx driver. 

## Why/what?

<img width="1899" height="1092" alt="image" src="https://github.com/user-attachments/assets/c312c4dc-15bb-4d42-8993-7d0c08bb8a52" />  
  
  
  
The UW undergraduate neuroscience program historically uses an A-M Systems 3000 AC/DC Differential Amplifier, a Humbug 60 Hz Noise Eliminator, and an AD Instruments PowerLab 4/26 Digitizer in concordance with LabChart 7 to record and analyze signals recorded from different model organisms. By doing so, students can learn about different information processing modalities in these model organisms to build neurobiology fundamentals. However, this system is dated and difficult to use on different operating systems. neusci-daq-visualizer is a flexible Python-based visualizer to transform signals into a computer-interpretable form and visualize it to make data analysis easy and efficient. We aim to cut out the PowerLab (which we are replacing with an NI USB-60030 and LabChart dependence with our tool.

## Install
Requires Python 3.11+. For recording, install the
[NI-DAQmx driver](https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html) first.
 
```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[nidaqmx,dev]"
```
 
## Run
```
neusci-daq -c config.toml                        # with the DAQ
neusci-daq -c config.toml --backend simulated    # without hardware
```
Edit `config.toml` to set the device, channels, sample rate and scaling.
 
To try review mode with sample data: `python examples/make_demo_session.py demo`, then
**File › Open…** the `demo` folder.

## Who

Made for and by students from UW's neuroscience major!
