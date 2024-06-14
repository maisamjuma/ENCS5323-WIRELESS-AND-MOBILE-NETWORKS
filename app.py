from flask import Flask, render_template, request

app = Flask(__name__)

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

        # Calculate maximum transmission rate for 4 parallel resource blocks per user
        max_transmission_rate = (4 * bits_per_ofdm_resource_block) / (resource_block_duration * 1000)  # convert ms to seconds

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

@app.route('/page4')
def page4():
    return render_template('page4.html')

@app.route('/page5')
def page5():
    return render_template('page5.html')

if __name__ == '__main__':
    app.run(debug=True)
