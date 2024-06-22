from flask import Flask, render_template, request
from math import log10, sqrt
import math
from scipy.special import factorial

app = Flask(__name__)

# Erlang B table (blocked calls cleared) with traffic intensities for given blocking probabilities
erlang_b_table = {
    0.001: [0.001, 0.046, 0.194, 0.439, 0.762, 1.1, 1.6, 2.1, 2.6, 3.1, 3.7, 4.2, 4.8, 5.4, 6.1, 6.7, 7.4, 8.0, 8.7, 9.4, 10.1, 10.8, 11.5, 12.2, 13.0, 13.7, 14.4, 15.2, 15.9, 16.7, 17.4, 18.2, 19.0, 19.7, 20.5, 21.3, 22.1, 22.9, 23.7, 24.4, 25.2, 26.0, 26.8, 27.6, 28.4],
    0.002: [0.002, 0.065, 0.249, 0.535, 0.900, 1.3, 1.8, 2.3, 2.9, 3.4, 4.0, 4.6, 5.3, 5.9, 6.6, 7.3, 7.9, 8.6, 9.4, 10.1, 10.8, 11.5, 12.3, 13.0, 13.8, 14.5, 15.3, 16.1, 16.8, 17.6, 18.4, 19.2, 20.0, 20.8, 21.6, 22.4, 23.2, 24.0, 24.8, 25.6, 26.4, 27.2, 28.1, 28.9, 29.7],
    0.005: [0.005, 0.105, 0.349, 0.701, 1.132, 1.6, 2.2, 2.7, 3.3, 4.0, 4.6, 5.3, 6.0, 6.7, 7.4, 8.1, 8.8, 9.6, 10.3, 11.1, 11.9, 12.6, 13.4, 14.2, 15.0, 15.8, 16.6, 17.4, 18.2, 19.0, 19.9, 20.7, 21.5, 22.3, 23.2, 24.0, 24.8, 25.7, 26.5, 27.4, 28.2, 29.1, 29.9, 30.8, 31.7],
    0.01: [0.010, 0.153, 0.455, 0.869, 1.361, 1.9, 2.5, 3.1, 3.8, 4.5, 5.2, 5.9, 6.6, 7.4, 8.1, 8.9, 9.7, 10.4, 11.2, 12.0, 12.8, 13.7, 14.5, 15.3, 16.1, 17.0, 17.8, 18.6, 19.5, 20.3, 21.2, 22.0, 22.9, 23.8, 24.6, 25.5, 26.4, 27.3, 28.1, 29.0, 29.9, 30.8, 31.7, 32.5, 33.4],
    0.012: [0.012, 0.168, 0.489, 0.922, 1.431, 2.0, 2.6, 3.2, 3.9, 4.6, 5.3, 6.1, 6.8, 7.6, 8.3, 9.1, 9.9, 10.7, 11.5, 12.3, 13.1, 14.0, 14.8, 15.6, 16.5, 17.3, 18.2, 19.0, 19.9, 20.7, 21.6, 22.5, 23.3, 24.2, 25.1, 26.0, 26.8, 27.7, 28.6, 29.5, 30.4, 31.3, 32.2, 33.1, 34.0],
    0.013: [0.013, 0.176, 0.505, 0.946, 1.464, 2.0, 2.7, 3.3, 4.0, 4.7, 5.4, 6.1, 6.9, 7.7, 8.4, 9.2, 10.0, 10.8, 11.6, 12.4, 13.3, 14.1, 14.9, 15.8, 16.6, 17.5, 18.3, 19.2, 20.0, 20.9, 21.8, 22.6, 23.5, 24.4, 25.3, 26.2, 27.0, 27.9, 28.8, 29.7, 30.6, 31.5, 32.4, 33.3, 34.2],
    0.015: [0.015, 0.19, 0.53, 0.99, 1.52, 2.1, 2.7, 3.4, 4.1, 4.8, 5.5, 6.3, 7.0, 7.8, 8.6, 9.4, 10.2, 11.0, 11.8, 12.6, 13.5, 14.3, 15.2, 16.0, 16.9, 17.7, 18.6, 19.5, 20.3, 21.2, 22.1, 22.9, 23.8, 24.7, 25.6, 26.5, 27.4, 28.3, 29.2, 30.1, 31.0, 31.9, 32.8, 33.7, 34.6],
    0.02: [0.020, 0.223, 0.602, 1.092, 1.657, 2.3, 2.9, 3.6, 4.3, 5.1, 5.8, 6.6, 7.4, 8.2, 9.0, 9.8, 10.7, 11.5, 12.3, 13.2, 14.0, 14.9, 15.8, 16.6, 17.5, 18.4, 19.3, 20.2, 21.0, 21.9, 22.8, 23.7, 24.6, 25.5, 26.4, 27.3, 28.3, 29.2, 30.1, 31.0, 31.9, 32.8, 33.7, 34.6, 35.6],
    0.03: [0.031, 0.282, 0.715, 1.259, 1.875, 2.5, 3.2, 4.0, 4.7, 5.5, 6.3, 7.1, 8.0, 8.8, 9.6, 10.4, 11.3, 12.2, 13.0, 13.9, 14.8, 15.7, 16.6, 17.5, 18.4, 19.3, 20.2, 21.1, 22.0, 22.9, 23.8, 24.8, 25.7, 26.6, 27.6, 28.5, 29.4, 30.4, 31.3, 32.2, 33.2, 34.1, 35.1, 36.0, 37.0],
    0.05: [0.052, 0.38, 0.922, 1.55, 2.25, 2.9, 3.7, 4.5, 5.3, 6.1, 6.9, 7.7, 8.6, 9.4, 10.3, 11.2, 12.1, 12.9, 13.8, 14.7, 15.6, 16.5, 17.4, 18.3, 19.2, 20.1, 21.0, 22.0, 22.9, 23.8, 24.7, 25.6, 26.6, 27.5, 28.5, 29.4, 30.3, 31.3, 32.2, 33.2, 34.1, 35.1, 36.0, 37.0, 38.0]
}

# Function to calculate the maximum distance
def calculate_max_distance(P0_dB, receiver_sensitivity, path_loss_exponent, reference_distance):
    P0 = 10 ** (P0_dB / 10)
    max_distance = reference_distance * (P0 / receiver_sensitivity) ** (1 / path_loss_exponent)
    return max_distance

# Function to calculate the maximum cell size (assuming hexagonal cells)
def calculate_max_cell_size(max_distance):
    return (3 * sqrt(3) / 2) * (max_distance ** 2)

# Function to calculate the number of cells in the service area
def calculate_number_of_cells(city_area, cell_area):
    return city_area / cell_area

# Function to calculate the traffic load in the whole system in Erlangs
def calculate_traffic_load(subscribers, average_calls_per_day, average_call_duration):
    return (subscribers * average_calls_per_day * average_call_duration) / (24 * 60)

# Function to calculate the traffic load in each cell in Erlangs
def calculate_traffic_load_per_cell(total_traffic_load, num_cells):
    return total_traffic_load / num_cells

#-----------------------------------------------
K = 1.38e-23  # Boltzmann constant in unitless
K_dB = 10 * log10(K)  # Convert Boltzmann constant to dB

# Lookup table for Eb/No based on modulation type and bit error rate
eb_no_lookup = {
    "BPSK/QPSK": {
        "10^-1": 0,
        "10^-2": 4,
        "10^-3": 7,
        "10^-4": 8.3,
        "10^-5": 9.5,
        "10^-6": 10.6,
        "10^-7": 11.5,
        "10^-8": 12
    },
    "8-PSK": {
        "10^-1": 0,
        "10^-2": 6.5,
        "10^-3": 10,
        "10^-4": 12,
        "10^-5": 13.1,
        "10^-6": 14,
        "10^-7": 14.7,
        "10^-8": 15.2
    },
    "16-PSK": {
        "10^-1": 0,
        "10^-2": 11.1,
        "10^-3": 14.1,
        "10^-4": 16,
        "10^-5": 17.2,
        "10^-6": 18.1,
        "10^-7": 18.8,
        "10^-8": 19.3
    }
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        # Get input values from the form
        bw = float(request.form['bw'])
        num_bits_quantizer = int(request.form['num_bits_quantizer'])
        source_encoder_rate = float(request.form['source_encoder_rate'])
        channel_encoder_rate = float(request.form['channel_encoder_rate'])
        interleaver_bits = int(request.form['interleaver_bits'])

        # Perform calculations
        sampling_frequency = 2 * bw
        quantization_levels = 2 ** num_bits_quantizer
        num_bits_source_input = 2 * bw * num_bits_quantizer
        source_encoder_output = num_bits_source_input * source_encoder_rate
        channel_encoder_output = source_encoder_output / channel_encoder_rate
        interleaver_output = channel_encoder_output

        # Return the results to the template
        return render_template('page1.html', 
                               sampling_frequency=sampling_frequency,
                               quantization_levels=quantization_levels,
                               num_bits_source_input=num_bits_source_input,
                               source_encoder_output=source_encoder_output,
                               channel_encoder_output=channel_encoder_output,
                               interleaver_output=interleaver_output)
    return render_template('page1.html')

@app.route('/page2', methods=['GET', 'POST'])
def page2():
    if request.method == 'POST':
        # Get input values from the form
        subcarrier_spacing = float(request.form['subcarrier_spacing'])
        num_ofdm_symbols = int(request.form['num_ofdm_symbols'])
        resource_block_duration = float(request.form['resource_block_duration'])
        modulation_scheme = request.form['modulation_scheme']
        num_parallel_resource_blocks = int(request.form['num_parallel_resource_blocks'])

        # Constants
        bandwidth_per_resource_block = 180  # kHz

        # Calculate number of bits per resource element based on modulation scheme
        if modulation_scheme == 'QPSK':
            bits_per_resource_element = 2
        elif modulation_scheme == '16-QAM':
            bits_per_resource_element = 4
        elif modulation_scheme == '64-QAM':
            bits_per_resource_element = 6
        elif modulation_scheme == '256-QAM':
            bits_per_resource_element = 8
        elif modulation_scheme == '1024-QAM':
            bits_per_resource_element = 10
        else:
            return render_template('page2.html', error="Invalid modulation scheme selected")

        # Calculate number of bits per OFDM symbol
        bits_per_ofdm_symbol = bits_per_resource_element * (bandwidth_per_resource_block / subcarrier_spacing)

        # Calculate number of bits per OFDM resource block
        bits_per_ofdm_resource_block = bits_per_ofdm_symbol * num_ofdm_symbols

        # Calculate maximum transmission rate for the given number of parallel resource blocks per user
        max_transmission_rate = (num_parallel_resource_blocks * bits_per_ofdm_resource_block) / (resource_block_duration * 1000)  # convert ms to seconds

        # Return the results to the template
        return render_template('page2.html', 
                               bits_per_resource_element=bits_per_resource_element,
                               bits_per_ofdm_symbol=bits_per_ofdm_symbol,
                               bits_per_ofdm_resource_block=bits_per_ofdm_resource_block,
                               max_transmission_rate=max_transmission_rate)
    
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')

@app.route('/calculate_db', methods=['GET', 'POST'])
def calculate_db():
    if request.method == 'POST':
        path_loss = float(request.form['path_loss'])
        frequency = float(request.form['frequency'])
        transmit_antenna_gain = float(request.form['transmit_antenna_gain'])
        receive_antenna_gain = float(request.form['receive_antenna_gain'])
        data_rate = float(request.form['data_rate'])
        antenna_feed_line_loss = float(request.form['antenna_feed_line_loss'])
        other_losses = float(request.form['other_losses'])
        fade_margin = float(request.form['fade_margin'])
        receiver_amplifier_gain = float(request.form['receiver_amplifier_gain'])
        noise_figure = float(request.form['noise_figure'])
        noise_temperature = float(request.form['noise_temperature'])
        link_margin = float(request.form['link_margin'])
        modulation_type = request.form['modulation_type']
        bit_error_rate = request.form['bit_error_rate']

        # Get Eb/No value from the lookup table
        eb_no_dB = eb_no_lookup[modulation_type][bit_error_rate]

        # Calculate power received in dB
        power_received_dB = link_margin + K_dB + noise_temperature + noise_figure + data_rate + eb_no_dB
        # Convert power received to unitless
        power_received_unitless = 10 ** (power_received_dB / 10)

        # Calculate total transmit power in dB
        total_transmit_power_dB = (power_received_dB + path_loss + antenna_feed_line_loss + other_losses + fade_margin) - (transmit_antenna_gain + receive_antenna_gain + receiver_amplifier_gain)
        # Convert total transmit power to unitless
        total_transmit_power_unitless = 10 ** (total_transmit_power_dB / 10)

        result = {
            "power_received_dB": power_received_dB,
            "power_received_unitless": power_received_unitless,
            "total_transmit_power_dB": total_transmit_power_dB,
            "total_transmit_power_unitless": total_transmit_power_unitless
        }

        return render_template('calculate_db.html', result=result)

    return render_template('calculate_db.html')

@app.route('/calculate_unitless', methods=['GET', 'POST'])
def calculate_unitless():
    if request.method == 'POST':
        path_loss = float(request.form['path_loss'])
        frequency = float(request.form['frequency']) * 1e6  # Convert MHz to Hz
        transmit_antenna_gain = float(request.form['transmit_antenna_gain'])
        receive_antenna_gain = float(request.form['receive_antenna_gain'])
        data_rate = float(request.form['data_rate']) * 1e3  # Convert kbps to bps
        antenna_feed_line_loss = float(request.form['antenna_feed_line_loss'])
        other_losses = float(request.form['other_losses'])
        fade_margin = float(request.form['fade_margin'])
        receiver_amplifier_gain = float(request.form['receiver_amplifier_gain'])
        noise_figure = float(request.form['noise_figure'])
        noise_temperature = float(request.form['noise_temperature'])
        link_margin = float(request.form['link_margin'])
        modulation_type = request.form['modulation_type']
        bit_error_rate = request.form['bit_error_rate']

        # Get Eb/No value from the lookup table
        eb_no_dB = eb_no_lookup[modulation_type][bit_error_rate]

        # Calculate Eb/No in unitless (linear scale)
        eb_no_unitless = 10 ** (eb_no_dB / 10)

        # Calculate power received in unitless (Watt)
        power_received_unitless = (link_margin * K * noise_temperature * noise_figure * data_rate * eb_no_unitless) / frequency

        # Convert power received to dB
        power_received_dB = 10 * log10(power_received_unitless)

        # Calculate total transmit power in unitless (Watt)
        total_transmit_power_unitless = (power_received_unitless * path_loss * antenna_feed_line_loss * other_losses * fade_margin) / (transmit_antenna_gain * receive_antenna_gain * receiver_amplifier_gain)

        # Convert total transmit power to dB
        total_transmit_power_dB = 10 * log10(total_transmit_power_unitless)

        result = {
            "power_received_unitless": power_received_unitless,
            "power_received_dB": power_received_dB,
            "total_transmit_power_unitless": total_transmit_power_unitless,
            "total_transmit_power_dB": total_transmit_power_dB
        }

        return render_template('calculate_unitless.html', result=result)

    return render_template('calculate_unitless.html')

@app.route('/page4', methods=['GET', 'POST'])
def page4():
    if request.method == 'POST':
        # Get input values from the form
        frame_size = float(request.form['frame_size'])
        rate = float(request.form['rate'])
        frame_rate = float(request.form['frame_rate'])
        propagation_time = float(request.form['propagation_time'])
        slotted_access = request.form['access_method']

        # Calculate frame period
        T_frame = frame_size / rate

        # Calculate load (G)
        G = frame_rate * T_frame

        # Calculate alpha (α)
        alpha = propagation_time / T_frame

        if slotted_access == 'slotted':
            # Calculate throughput for slotted access
            numerator =  alpha * G * math.exp(-2 * alpha * T_frame)
            denominator = G * (1 +  alpha - math.exp(-2 * alpha * G))
            throughput = numerator / denominator
        elif slotted_access == 'non-slotted':
            # Calculate throughput for non-slotted access
            numerator = G * math.exp(-2 * alpha * T_frame)
            denominator = G * (1 + 2 * alpha) + math.exp(-alpha * G)
            throughput = numerator / denominator
        else:
            return render_template('page4.html', error="Invalid access method selected")

        # Calculate throughput as percentage
        throughput_percent = throughput * 100
        
        return render_template('page4.html', throughput_percent=throughput_percent)
    
    return render_template('page4.html')

@app.route('/page5', methods=['GET', 'POST'])
def page5():
    if request.method == 'POST':
        # Get input values from the form
        SIR_dB = float(request.form['SIR_dB'])
        P0_dB = float(request.form['P0_dB'])
        reference_distance = float(request.form['reference_distance'])
        path_loss_exponent = float(request.form['path_loss_exponent'])
        receiver_sensitivity = float(request.form['receiver_sensitivity'])
        city_area = float(request.form['city_area'])
        subscribers = float(request.form['subscribers'])
        average_calls_per_day = float(request.form['average_calls_per_day'])
        average_call_duration = float(request.form['average_call_duration'])
        num_timeslots = float(request.form['num_timeslots'])
        GOS = float(request.form['GOS'])
        num_co_channel_cells = float(request.form['num_co_channel_cells'])

        # Perform calculations
        max_distance = calculate_max_distance(P0_dB, receiver_sensitivity, path_loss_exponent, reference_distance)
        max_cell_size = calculate_max_cell_size(max_distance)
        num_cells = math.ceil(calculate_number_of_cells(city_area, max_cell_size))
        total_traffic_load = calculate_traffic_load(subscribers, average_calls_per_day, average_call_duration)
        traffic_load_per_cell = calculate_traffic_load_per_cell(total_traffic_load, num_cells)

        print(total_traffic_load)
        # Function to calculate the Erlang B formula using an iterative approach
        def erlang_b(traffic_load_per_cell, channels):
            invB = 1.0
            for i in range(1, channels + 1):
                invB = 1.0 + invB * i / traffic_load_per_cell
            return 1.0 / invB

        # Get user input for traffic intensity and blocking probability
        while True:
            try:
                if GOS not in erlang_b_table:
                    raise ValueError("Blocking probability not in the table")
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")

        # Find the number of channels needed to achieve the given blocking probability
        channels = 0
        while True:
            channels += 1
            current_b = erlang_b(traffic_load_per_cell, channels)
            if current_b <= GOS:
                break
    
        # Calculate the maximum allowable traffic intensity from the table for the given blocking probability
        max_traffic_intensity = erlang_b_table[GOS][channels - 1]


        min_carriers = max_traffic_intensity

        # Calculate SIR in unitless
        SIR_unitless = 10 ** (SIR_dB / 10)

        # Calculate number of cells in each cluster
        h = average_call_duration
        num_cells_per_cluster = math.ceil(((SIR_unitless * num_co_channel_cells) ** (1 / h)) ** 2 / 3)

        result = {
            "max_distance": max_distance,
            "max_cell_size": max_cell_size,
            "num_cells": num_cells,
            "total_traffic_load": total_traffic_load,
            "traffic_load_per_cell": traffic_load_per_cell,
            "num_cells_per_cluster": num_cells_per_cluster,
            "min_carriers": min_carriers,
        }

        return render_template('page5.html', result=result)

    return render_template('page5.html')




if __name__ == '__main__':
    app.run(debug=True)