import time
import requests
import logging
import numpy as np
from datetime import datetime, timedelta
from PocketfulAPI.pocketfulwebsocket import PocketfulSocket


#1. Fetch data every 3 minutes to get the latest candles and calculate RSI.
#2. Check RSI condition – For SELL: RSI crossed 70 once and is now below 70.
#                       – For BUY: RSI crossed 30 once and is now above 30.
#3. Check candle conditions – Analyze the last three candles to match your bullish/bearish pattern.
#4. Monitor LTP every 5 seconds – Once the conditions are met, start checking the Last Traded Price (LTP).
#5. Trigger order – If LTP falls below the low of the bullish candle, print the trigger order with stop loss and target.

def establish_websocket_connection(clientId, access_token, exchange_code, instrument_token):
    """
    Establish WebSocket connection and subscribe to market data.
    
    Args:
        clientId (str): Client ID for authentication
        access_token (str): Access token for authentication
        exchange_code (int): Exchange code for the instrument
        instrument_token (int): Specific instrument token to subscribe
    
    Returns:
        tuple: (connection object, boolean indicating connection status)
    """
    try:
        # Create WebSocket connection
        conn = PocketfulSocket(clientId, access_token)
        
        # Establish WebSocket connection
        ws_status = conn.run_socket()
        if not ws_status:
            print("WebSocket connection failed!")
            return None, False
        print("WebSocket connected successfully!")
        
        # Subscribe to market data
        marketdata_payload = {
            'exchangeCode': exchange_code, 
            'instrumentToken': instrument_token
        }
        conn.subscribe_detailed_marketdata(marketdata_payload)
        print("Subscribed to market data.")
        
        return conn, True
    
    except Exception as e:
        print(f"Error setting up WebSocket: {e}")
        return None, False   


def fetch_chart_data(start_time, end_time, clientId, access_token):
    url = 'https://web.pocketful.in/api/space/v1/screeners/chartData'
    headers = {
        'accept': 'application/json, text/plain, */*',
        'authorization': 'Bearer ' + access_token,
        'clientid': clientId,
        'content-type': 'application/json',
        'p-deviceid': 'ebcca4f9-3073-40de-b4ce-f444cc77430c',
        'p-devicetype': 'web',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/133.0.0.0 Safari/537.36'
    }

    payload = {
        'candleType': '1',
        'endTime': str(end_time),
        'exchange': 'NSE_INDICES',
        'startTime': str(start_time),
        'token': '26000',
        'dataDuration': '3'
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data['data']['data']['candles']
    else:
        print("Error fetching data:", response.status_code, response.text)
        return []


def calculate_rsi(candles, period=14):
    """
    Calculate Relative Strength Index (RSI) from candle data.
    
    Parameters:
    candles - List of candle data where each candle is 
              [timestamp, open, high, low, close, volume]
    period - RSI period (default: 14)
    
    Returns:
    List of [timestamp, rsi_value] pairs
    """

    #1. Extract closing prices from candles
    closing_prices = [float(candle[4]) for candle in candles]

    #2. Calculate price changes
    price_diff = np.diff(closing_prices)
    
    gains = np.copy(price_diff)
    losses = np.copy(price_diff)
    
    # Separate gains and losses
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)

    #4. Initialize result list and Check Data Sufficiency
    rsi_values = []
    
    if len(closing_prices) <= period:
        return rsi_values    

    #5. Calculate first average gain and loss
    avg_gain = np.sum(gains[:period]) / period
    avg_loss = np.sum(losses[:period]) / period

    #explain: This calculates the average gain and loss over the first period data points.
    # If period = 14 and we have gains and losses for the first 14 periods:
    # avg_gain = (sum of all gains in first 14 periods) / 14
    # avg_loss = (sum of all losses in first 14 periods) / 14
    

    #6. Add None for periods where RSI can't be calculated yet (Add Placeholder Values)
    for i in range(period):
        rsi_values.append([candles[i][0], None])
    
    #7. Calculate first RSI
    if avg_loss == 0:
        rs = float('inf')
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    #explain: This calculates the first RSI value using the formula:
    # RS (Relative Strength) = Average Gain / Average Loss
    # RSI = 100 - (100 / (1 + RS))

    rsi_values.append([candles[period][0], rsi])
    
    #8. Calculate remaining RSI values using smoothed averages
    for i in range(period + 1, len(closing_prices)):
        # Update average gain and loss using smoothing formula
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        
        if avg_loss == 0:
            rs = float('inf')
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append([candles[i][0], round(rsi, 2)])


    #explain: This uses the smoothing technique to calculate the remaining RSI values:
    # New Average Gain = ((Previous Average Gain) * (period - 1) + Current Gain) / period
    # New Average Loss = ((Previous Average Loss) * (period - 1) + Current Loss) / period
    return rsi_values
 

# Check conditions for trigger
def check_strategy(clientId, access_token, exchange_code, instrument_token):
    rsi_crossed_70 = False  # Flag to track if RSI crossed 70 once
    rsi_crossed_30 = False  # Flag to track if RSI crossed 30 once

    # set start and end time (start time will be yesterday 9:15 and end time will be current time)  
    start_time = get_unix_time_for_specific_time(9,15,-1)
    end_time = int(time.time())
    
    log_event("Starting the strategy...")

    #1 check until RSI crosses 70 or 30 once and fell below/rose above accordingly
    while True:

        # validations ################################################################################################

        current_time = int(time.time())
        
        # Ensure API calls are spaced exactly 3 minutes apart
        if current_time < end_time:            
            minutes, seconds = divmod(end_time - current_time, 60)
            log_event("Next candle will be fetched in {} minutes {} seconds.".format(minutes, seconds))        
            time.sleep(end_time - current_time)  # Wait until the next scheduled call
            continue
            
        # if todays time become past 2pm exit the loop
        if datetime.fromtimestamp(end_time).hour >= 16:
            log_event("Time is past 2pm. Exiting the loop.")
            break

        # Fetch candle data every 3 minutes
        candles = fetch_chart_data(start_time, end_time, clientId, access_token)

        # Calculate RSI based on candles data
        rsi_values = calculate_rsi(candles)
        latest_rsi = rsi_values[-1][1] if rsi_values else 0

         # Check if RSI crossed 70 or 30 today
        if not rsi_crossed_70:
            rsi_crossed_70 = any(rsi is not None and rsi > 70 for _, rsi in rsi_values)
        
        if not rsi_crossed_30:
            rsi_crossed_30 = any(rsi is not None and rsi < 30 for _, rsi in rsi_values)
            
        log_event(f"Latest RSI: {latest_rsi}, RSI crossed 70: {rsi_crossed_70}, RSI crossed 30: {rsi_crossed_30}")


        # Determine strategy based on RSI conditions
        if rsi_crossed_70 and latest_rsi < 70:
            strategy_type = "SELL"
            log_event("RSI has crossed 70 and is now below 70. Using SELL strategy.")
        elif rsi_crossed_30 and latest_rsi > 30:
            strategy_type = "BUY"
            log_event("RSI has crossed 30 and is now above 30. Using BUY strategy.")
        else:
            strategy_type = None
            log_event("Waiting for RSI conditions to be met...")
            end_time += 180
            continue

        
        # calulations #######################################################################


        # Get the last three candles
        candle_current = candles[-1]
        candle_prev = candles[-2]
        candle_prev2 = candles[-3]
        candle_start = candle_prev2

        # Extract candle details for better clarity
        current_time, open_current, high_current, low_current, close_current = candle_current[:5]
        prev_time, open_prev, high_prev, low_prev, close_prev = candle_prev[:5]
        prev2_time, open_prev2, high_prev2, low_prev2, close_prev2 = candle_prev2[:5]       
        high_start = candle_start[2]
        low_start = candle_start[3]

        pattern_matched = False

         # Check pattern based on strategy type
        if strategy_type == "SELL":

            # Condition: current candle is bullish candle (green candle)
            is_current_bullish = close_current > open_current 
            # Condition: previous two candle is bearish candle (red candle)
            is_contiguous_red = close_prev < open_prev and close_prev2 < open_prev2
            # Condition: both previous two candle is bearish candle (red candle) are not overlapping (inside candles)
            is_prev_lower_than_prev2 = low_prev2 > low_prev  
            # condition: prev_high should not cross prev2_High
            is_prev1_high_not_crossed_prev2 = high_prev2 > high_prev
            # Condition: current high not crossing 1st red candle high
            is_current_below_prev2_high = high_current <= high_prev2 
            # condition : drop in 2nd (prev1) candle should be less than 40
            is_prev_candle_drop_less_than_40 = high_prev - low_prev < 40        

            # if all conditions match then start LTP monitoring
            pattern_matched = (is_current_bullish and is_contiguous_red and is_prev_lower_than_prev2 and 
                              is_current_below_prev2_high and is_prev_candle_drop_less_than_40 and 
                              is_prev1_high_not_crossed_prev2)

        elif strategy_type == "BUY":
            # Condition: current candle is bearish candle (red candle)
            is_current_bearish = close_current < open_current 
            # Condition: previous two candle is bullish candle (green candle)
            is_contiguous_green = close_prev > open_prev and close_prev2 > open_prev2
            # Condition: both previous two candle is bullish candle (green candle) are not overlapping (inside candles)
            is_prev_greater_than_prev2 = high_prev2 < high_prev  
            # Condition: current low not crossing 1st green candle low
            is_current_above_prev2_low = low_current >= low_prev2 
            # condition : gain in 2nd (prev1) candle should be less than 40
            is_prev_candle_gain_less_than_40 = high_prev - low_prev < 40

            # if all conditions match then start LTP monitoring
            pattern_matched = (is_current_bearish and is_contiguous_green and is_prev_greater_than_prev2 and 
                              is_current_above_prev2_low and is_prev_candle_gain_less_than_40)


        # if all conditions matches then start LTP monitoring
        if pattern_matched:
            
            log_event("✅ Pattern Matched! Starting LTP monitoring...")

            #setup webhook connection
            conn, connection_status = establish_websocket_connection(
                    clientId, access_token, exchange_code, instrument_token
            )

            # Exit if connection failed
            if not connection_status:
                print("Failed to establish WebSocket connection. Exiting strategy.")
                return
                                                
            # iterate until ltp did not achive target or break stop loss
            while True:

                # Set stop loss and target based on strategy type
                if strategy_type == "SELL":
                    stop_loss = max(high_prev, high_current)
                    trigger_condition = lambda ltp: ltp < low_current
                    pattern_invalidation = lambda ltp: ltp > high_start
                else:  # BUY strategy
                    stop_loss = min(low_prev, low_current)
                    trigger_condition = lambda ltp: ltp > high_current
                    pattern_invalidation = lambda ltp: ltp < low_start

                # Target is always double the stop loss (relative to entry)
                target = 2 * stop_loss if strategy_type == "SELL" else stop_loss / 2

                next_candle_time = end_time + 180  # Next candle will be in 3 minutes
                found_result = False


                # listen to ltp till next candle comes (after 3 min here).
                while int(time.time()) < next_candle_time: 
                    log_event("Our SL: {}, Target: {}".format(stop_loss, target))                   

                    # get market data to fetch ltp
                    detailed_market_data = conn.read_detailed_marketdata()
                    ltp = detailed_market_data.get('last_traded_price', 0)/100            
                    
                    log_event(f"LTP .....: {ltp}")

                    #need to check this condition 'ltp < low_current' untill next 3 min 

                    # Check if trigger condition is met
                    if trigger_condition(ltp):                                   
                        log_event(f"Trigger Order: {strategy_type} at {ltp}, SL: {stop_loss}, Target: {target}")                    

                        # monitor ltp till stop loss or target
                        while True:
                            detailed_market_data = conn.read_detailed_marketdata()
                            ltp = detailed_market_data.get('last_traded_price', 0)
                            log_event(f"Ltp .....: {ltp}")

                            # Check stop loss based on strategy
                            if (strategy_type == "SELL" and ltp > stop_loss) or (strategy_type == "BUY" and ltp < stop_loss):
                                found_result = True
                                log_event(f"❌ Stop Loss Hit on LTP: {ltp}, SL: {stop_loss} ❌")
                                break

                            # Check target based on strategy
                            elif (strategy_type == "SELL" and ltp <= target) or (strategy_type == "BUY" and ltp >= target):
                                found_result = True
                                log_event(f"✅ Target Achieved on LTP: {ltp} ✅")
                                end_time = next_candle_time
                                break


                            time.sleep(0.5)
                    
                    # break condition: if ltp crosses the high of the  starting candle, invalidate the pattern
                    elif pattern_invalidation(ltp):
                        found_result = True
                        log_event(f"Pattern invalidated ltp: {ltp} > high_start {high_start}. Restarting strategy...")
                        break

                    time.sleep(0.5)


                # if 3 minutes completed after pattern match 

                # if there is no result till yet
                if not found_result:
                    #fetch next candle, then upcoming candle become current, and current candle become previous
                    candles = fetch_chart_data(start_time, next_candle_time, clientId, access_token)
                    candle_current = candles[-1]
                    candle_prev = candles[-2]

                    # Update high_current and high_prev for stop loss based on strategy
                    if strategy_type == "SELL":
                        high_current = candle_current[2]
                        high_prev = candle_prev[2]
                    else:  # BUY strategy
                        low_current = candle_current[3]
                        low_prev = candle_prev[3]

                    continue                  

                # if there is a result; break
                else:
                    end_time = next_candle_time
                    break                                      

        #else: condition does not matched                    
        else:
            log_event("Pattern not matched. Waiting for next check...⌚")
            end_time = end_time + 180
            continue


        #close the websocket connection, if open
        if connection_status:
            marketdata_payload = {
                'exchangeCode': exchange_code, 
                'instrumentToken': instrument_token
            }
            log_event("Closing WebSocket connection...")
            conn.unsubscribe_detailed_marketdata(marketdata_payload)


def get_unix_time_for_specific_time(hour, minute, days_ahead):
    """
    Get Unix timestamp for a specific time today or in future days
    
    Args:
    - hour (int): Hour of the day (0-23)
    - minute (int): Minute of the hour (0-59)
    - days_ahead (int): Number of days ahead (0 for today, 1 for tomorrow, etc.)
    
    Returns:
    - int: Unix timestamp for the specified time
    """
    # Get today's date
    target_date = datetime.now().date() + timedelta(days=days_ahead)
    
    # Create datetime object for the specific time
    target_datetime = datetime.combine(
        target_date, 
        datetime.min.time().replace(hour=hour, minute=minute)
    )
    
    # Convert to Unix timestamp
    return int(target_datetime.timestamp())


# Configure logging
logging.basicConfig(
    filename="strategy_log.txt",  # Log file name
    level=logging.INFO,           # Set logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format: timestamp - level - message
)


def log_event(message, level="info"):
    """Helper function to log messages at different levels."""
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)


if __name__ == '__main__':
   
   clientId = "PA0002"
   accessToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJibGFja2xpc3Rfa2V5IjoiUEEwMDAyOnFOZmhsQS8rU002ZzJEOU41em40VHciLCJjbGllbnRfaWQiOiJQQTAwMDIiLCJjbGllbnRfdG9rZW4iOiJRVEV5T0VkRFRRLjN6d0N5RWZuWHZ3bl9keEU3YWRRVFNfNTFHOC0yd01uMWxQMEg2d2JES1YzdW9abmNEZEFLQVYzQlRjLnNtblQ4aWhEOTIyNWlQZmcuR085RkFaTThBczNuY2JSZThPUUlhYWpZQ00waEhacDh3V3ZyTGo4NVZzdXJKbk0xeFE1ZFd4cUZvYlZUUUtHb0xITWx4LThLZThEM09TWU1hN1N3YWt4LXVWMnFPWHpRX1o0dlhoMXV3REJBRl9xejhtNWU1cmdmWkIxVmlvc0dDc0JMTWh1aGp5SS11S09SRTNHaC5ZNUw0M21ETnFvSVVFV09FWDJ5ckpBIiwiZGV2aWNlIjoid2ViIiwiZGV2aWNlX2lkIjpudWxsLCJpcCI6bnVsbCwiZXhwIjoxNzQzMjI1MjYyOTQxfQ.tj0yHV0Zjz7CgGshrsUw22llgHfNXYk_2gARgkw3S20"
   exchangeCode = 1
   instrumentToken = 26000
   log_event("-------------------------started-------------------------------")
   check_strategy(clientId, accessToken, exchangeCode, instrumentToken)