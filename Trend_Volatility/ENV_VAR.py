# API
APCA_API_BASE_URL='https://paper-api.alpaca.markets'
APCA_API_KEY_ID='PK1XMBTCP6J5TXXLM3VV'
APCA_API_SECRET_KEY='Ff36rBTDi3HVfwr/UI64Xkq5BsPSFMSXA1PcYlg4'
AV_API_KEY_ID='F1QACIJR7DUPCA2T'
APCA_MAX_CALLS = 200
AV_MAX_CALLS = 5

# Indices for data
OPEN_INDEX = 0
HIGH_INDEX = 1
LOW_INDEX = 2
CLOSE_INDEX = 3
VOLUME_INDEX = 4
METRICS = ['Open','High','Low','Close','Volume']

# For historical data (Set end to something very close to current day, like a week before)
DATE_START = '2016-01-01'
DATE_END = '2020-08-05'

# Default tolerances
DEFAULT_TOLS = (0.01,-0.01)
