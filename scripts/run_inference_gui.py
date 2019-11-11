import PySimpleGUI as sg

# sg.ChangeLookAndFeel('GreenTan')
# 
# column1 = [[sg.Text('Column 1', background_color='#d3dfda', justification='center', size=(10, 1))],      
#            [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 1')],      
#            [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 2')],      
#            [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 3')]]      
# layout = [      
#     [sg.Text('All graphic widgets in one window!', size=(30, 1), font=("Helvetica", 25))],      
#     [sg.Text('Here is some text.... and a place to enter text')],      
#     [sg.InputText('This is my text')],      
#     [sg.Checkbox('My first checkbox!'), sg.Checkbox('My second checkbox!', default=True)],      
#     [sg.Radio('My first Radio!     ', "RADIO1", default=True), sg.Radio('My second Radio!', "RADIO1")],      
#     [sg.Multiline(default_text='This is the default Text should you decide not to type anything', size=(35, 3)),      
#      sg.Multiline(default_text='A second multi-line', size=(35, 3))],      
#     [sg.InputCombo(('Combobox 1', 'Combobox 2'), size=(20, 3)),      
#      sg.Slider(range=(1, 100), orientation='h', size=(34, 20), default_value=85)],      
#     [sg.Listbox(values=('Listbox 1', 'Listbox 2', 'Listbox 3'), size=(30, 3)),      
#      sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=25),      
#      sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=75),      
#      sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=10),      
#      sg.Column(column1, background_color='#d3dfda')],      
#     [sg.Text('_'  * 80)],      
#     [sg.Text('Choose A Folder', size=(35, 1))],      
#     [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),      
#      sg.InputText('Default Folder'), sg.FolderBrowse()],      
#     [sg.Submit(), sg.Cancel()]      
# ]
# 
# window = sg.Window('Everything bagel', default_element_size=(40, 1)).Layout(layout)
# button, values = window.Read()
# sg.Popup(button, values)

sg.ChangeLookAndFeel('GreenTan')

column1 = [[sg.Text('Configuration files', background_color='#d3dfda', justification='center', size=(25, 1))],      
           [sg.Spin(values=('Standard (mouse)', 'High resolution (mouse)', 'Human'),
                    initial_value='Standard (mouse)', size=(20, 1))]]

layout = [
    [sg.Text('Lung Segmentation using CNN', size=(30, 1), font=("Helvetica", 25))],
    [sg.Column(column1, background_color='#d3dfda')],
    [sg.Submit(), sg.Cancel()]
]

window = sg.Window('SIENA', default_element_size=(40, 1)).Layout(layout)
button, values = window.Read()
# sg.Popup(button, values)
if values[0] == 'Standard (mouse)':
    frame_layout = [
        [sg.Checkbox('Cluster correction', change_submits = True, enable_events=True, default='0',key='print_output')],
        [sg.InputText(('Minimum extent'), size=(20, 3)),      
         sg.Slider(range=(1, 1000), orientation='h', size=(34, 20), default_value=350)]]
    layout = [
        [sg.Text('Prepareing the inference for standard mouse CT images', size=(30, 1), font=("Helvetica", 25))],
        [sg.Text('Here is some text.... and a place to enter text')],
        [sg.InputText('This is my text')],
        [sg.Checkbox('My first checkbox!'), sg.Checkbox('My second checkbox!', default=True)],
        [sg.Radio('My first Radio!     ', "RADIO1", default=True), sg.Radio('My second Radio!', "RADIO1")],      
        [sg.Multiline(default_text='This is the default Text should you decide not to type anything', size=(35, 3)),      
         sg.Multiline(default_text='A second multi-line', size=(35, 3))],      
        [sg.Frame('Nataveni', frame_layout, font='Any 12', title_color='black')],    
        [sg.Text('_'  * 80)],      
        [sg.Text('Choose A Folder', size=(35, 1))],      
        [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),      
         sg.InputText('Default Folder'), sg.FolderBrowse()],      
        [sg.Submit(), sg.Cancel()]      
    ]
    window = sg.Window('SIENA2', default_element_size=(40, 1)).Layout(layout)
    button, values = window.Read()
    sg.Popup(button, values)
