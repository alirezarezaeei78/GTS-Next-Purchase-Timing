"""Convert the official Online Retail II workbook to the preparation input format.

Download the workbook yourself under the source provider's terms. This utility
does not download, redistribute, filter, or model any records.
"""
from pathlib import Path
import argparse
import pandas as pd
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--online-retail-workbook',type=Path,required=True)
p.add_argument('--data-root',type=Path,required=True)
a=p.parse_args()
sheets=pd.read_excel(a.online_retail_workbook,sheet_name=None)
frame=pd.concat(list(sheets.values()),ignore_index=True)
expected={'Invoice','StockCode','Quantity','InvoiceDate','Price','Customer ID'}
assert expected<=set(frame),set(frame)
folder=a.data_root/'online_retail_ii';folder.mkdir(parents=True,exist_ok=True)
frame.to_parquet(folder/'online_retail_II.parquet',index=False)
print('Converted',len(frame),'unfiltered source rows. Clean filtering is performed by prepare.py.')
