from flask import Flask, render_template
import os
import liver_stats as lstat
#import 

print(os.path.abspath('./test.png'))

TEMPLATE_DIR = os.path.abspath('./templates')
STATIC_DIR = os.path.abspath('./static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.route('/')
def home():
   return render_template('home.html', value = '17/2_116.jpeg')

@app.route('/high_fibrosis_example/')
def high_segment():
   #fiber_raw, fiber_excess = lstat.raw_perc_fibrosis(image_path='./static/objC1_files', raw_img_path='./static/c1_vips_files')
   fiber_raw = f'{54.1} % fibrous tissue'
   fiber_excess = 'In Progress'
   return render_template('high_fibrosis_example.html',value=fiber_raw, value2=fiber_excess)


@app.route('/low_fibrosis_example/')
def low_segment():
   #fiber_raw, fiber_excess = lstat.raw_perc_fibrosis(image_path='./static/objC0_files', raw_img_path='./static/c0_vips_files')
   fiber_raw = f'{10.7} % fibrous tissue'
   fiber_excess = 'In Progress'
   return render_template('low_fibrosis_example.html',value=fiber_raw, value2=fiber_excess)


# @app.route('/dataTable/')
# def dataTable():
   
#    return render_template('dataTable.html')

if __name__ == '__main__':
   app.run()
