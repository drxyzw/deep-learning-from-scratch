from ch05.code_5_4_1_Layer import *

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 0.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_pretax_price_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
pretax_price = add_pretax_price_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(pretax_price, 1. + tax)
print(apple_price, orange_price, pretax_price, price)

# backward
dprice = 1.
dpretax_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_pretax_price_layer.backward(dpretax_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
print(dpretax_price, dtax, dapple_price, dorange_price, dapple, dapple_num, dorange, dorange_num)
