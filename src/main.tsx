import datetime

def get_bucket(product_name, date):
    operational_start_date = datetime.datetime(2025, 1, 1)  # Assuming this is when GOES-19 became operational

    if date > operational_start_date:
        return f"noaa-goes19/{product_name}"
    else:
        return f"noaa-goes16/{product_name}"

import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(<App />);

