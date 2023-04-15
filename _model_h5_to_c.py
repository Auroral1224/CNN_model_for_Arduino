from tinymlgen import port
from keras.models import load_model
model = load_model('non_pruned.h5')
if __name__ == '__main__':
    c_code = port(model, variable_name='digits_model',
                  pretty_print=True, optimize=False)

with open('non_pruned.h', 'w') as f:
    f.write(c_code)
