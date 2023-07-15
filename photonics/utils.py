from datetime import datetime

date_now = lambda: datetime.now().strftime('%Y%m%d')[2:]
time_now = lambda: datetime.now().strftime('%H%M')
datetime_now = lambda: date_now() + "_" + time_now()

make_fname = lambda n: f"/home/lab/asengupta/photonics/data/pl_{date_now()}/{n}_{datetime_now()}"