from flask import Flask, render_template, request
from math import log10, sqrt
import math
from scipy.special import factorial

app = Flask(__name__)

# Constants
SIR_dB = 13  # Minimum Signal to Interference Ratio in dB
P0_dB = -22  # Power measured at reference distance in dB
reference_distance = 10  # Reference distance in meters
path_loss_exponent = 3  # Path loss exponent for urban area
receiver_sensitivity = 7e-6  # Receiver sensitivity in watts
city_area = 4000000  # Area of the city in square meters
subscribers = 80000  # Number of subscribers
average_calls_per_day = 8  # Average number of calls per day per subscriber
average_call_duration = 3  # Average call duration in minutes
num_timeslots = 8  # Number of timeslots per carrier
GOS = 0.02  # Grade of Service (call drop probability)
num_co_channel_cells = 6  # Given: Number of co-channel interfering cells

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

# Function to calculate the minimum number of carriers needed to achieve the required QoS
def calculate_minimum_carriers(traffic_load_per_cell, num_timeslots, GOS):
    def erlang_b_formula(c, a):
        inv_b = sum((a ** i) / factorial(i) for i in range(c + 1))
        return (a ** c) / (factorial(c) * inv_b)

    c = 1
    while erlang_b_formula(c, traffic_load_per_cell) > GOS:
        c += 1
    return (c // num_timeslots) + 1

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
        min_carriers = math.ceil(calculate_minimum_carriers(traffic_load_per_cell, num_timeslots, GOS))
        min_carriers_QoS_0_05 = math.ceil(calculate_minimum_carriers(traffic_load_per_cell, num_timeslots, 0.05))

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
            "min_carriers_QoS_0_05": min_carriers_QoS_0_05
        }

        return render_template('page5.html', result=result)

    return render_template('page5.html')




if __name__ == '__main__':
    app.run(debug=True)