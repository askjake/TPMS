#!/bin/bash
cd ~/TPMS
source venv/bin/activate
echo "Starting TPMS Scanner on http://localhost:8502"
streamlit run app.py --server.port 8502
