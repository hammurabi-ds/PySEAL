import time
import random
import pickle
import numpy as np
import threading
import seal
from seal import ChooserEvaluator, \
	Ciphertext, \
	Decryptor, \
	Encryptor, \
	EncryptionParameters, \
	Evaluator, \
	IntegerEncoder, \
	FractionalEncoder, \
	KeyGenerator, \
	MemoryPoolHandle, \
	Plaintext, \
	SEALContext, \
	EvaluationKeys, \
	GaloisKeys, \
	PolyCRTBuilder, \
	ChooserEncoder, \
	ChooserEvaluator, \
	ChooserPoly


def mat_encrypt(matrix, encryption, encoding):
	result=np.zeros((matrix.shape[0], matrix.shape[1])).tolist()
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			plain = encoding.encode(matrix[i][j])
			encrypted = Ciphertext()
			encryption.encrypt(plain, encrypted)
			result[i][j] = encrypted
	
	return np.asarray(result)

def mat_decrypt(matrix, decryption, encoding):
	result=np.zeros((matrix.shape[0], matrix.shape[1])).tolist()
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			plain_result = Plaintext()
			decryption.decrypt(matrix[i][j], plain_result)
			result[i][j] = (str)(encoding.decode_int32(plain_result))
	
	return np.asarray(result)

def scal_encrypt(scal, encryption, encoding):
	plain = encoding.encode(scal)
	encrypted = Ciphertext()
	encryption.encrypt(plain, encrypted)
	return encrypted

def initialize_encryption():
	print_example_banner("Example: Basics I");
	parms = EncryptionParameters()
	parms.set_poly_modulus("1x^2048 + 1")
	# factor: 0xfffffffff00001.
	parms.set_coeff_modulus(seal.coeff_modulus_128(2048))
	parms.set_plain_modulus(1 << 8)
	context = SEALContext(parms)
	print_parameters(context);
	# Here we choose to create an IntegerEncoder with base b=2.
	encoder = IntegerEncoder(context.plain_modulus())
	keygen = KeyGenerator(context)
	public_key = keygen.public_key()
	secret_key = keygen.secret_key()
	encryptor = Encryptor(context, public_key)
	evaluator = Evaluator(context)
	decryptor = Decryptor(context, secret_key)
	return encryptor, evaluator, decryptor, encoder, context

# Matrix arithmetics
def mat_mulscal(matrix, scalar, evaluation):
	result=np.zeros((matrix.shape[0], matrix.shape[1])).tolist()
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			temp = Ciphertext()
			evaluation.multiply(matrix[i][j],scalar, temp)
			result[i][j] = temp
	return np.asarray(result)

def mat_sumscal(matrix,scalar,evaluation):
	result = np.zeros((matrix.shape[0], matrix.shape[1])).tolist()
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			temp = Ciphertext()
			evaluation.add(matrix[i][j],scalar,temp)
			result[i][j] = temp
	return np.asarray(result)

def sum_values(matrix, evaluation):
	result = 0
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			evaluation.add(result,matrix[i][j])
	return result

def mat_id(size, encryption, encoding):
	result = []
	for i in range(size):
		x = []
		for j in range(size):
			encrypted_data = Ciphertext()
			if (i==j):
				encryption.encrypt(encoding.encode(1), encrypted_data)
			else:
				encryption.encrypt(encoding.encode(0), encrypted_data)
			x.append(encrypted_data)
		result.append(x)
	return np.asarray(result)

def mat_sub(matrix_1, matrix_2, evaluation, encryption, encoding):
	result=np.zeros((matrix_1.shape[0], matrix_1.shape[1])).tolist()
	for i in range(matrix_1.shape[0]):
		for j in range(matrix_1.shape[1]):
			temp = Ciphertext()
			temp2 = Ciphertext()
			evaluation.multiply(matrix_2[i][j], scal_encrypt(-1, encryption, encoding ), temp)
			evaluation.add(matrix_1[i][j], temp, temp2)
			result[i][j] = temp2
	return np.asarray(result)

def mat_add(matrix_1, matrix_2, evaluation):
	result=np.zeros((matrix_1.shape[0], matrix_1.shape[1])).tolist()
	for i in range(matrix_1.shape[0]):
		for j in range(matrix_1.shape[1]):
			evaluation.add(matrix_1[i][j], matrix_2[i][j])
			result[i][j] = matrix_1[i][j]
	return np.asarray(result)

def create_matrix(n,m, number):
	result=np.zeros((n, m)).tolist()
	for i in range(n):
		for j in range(m):
			result[i][j]=number
	return np.asarray(result)

def mat_mul(m1,m2,evaluator,encr,enc):
    result = [] # final result
    for i in range(len(m1)):
        row = [] # the new row in new matrix
        for j in range(len(m2[0])):
            product = 0
            auxiliar = scal_encrypt(product,encr,enc)
            auxiliar2 = scal_encrypt(product,encr,enc)
            for v in range(len(m2[i])):
                evaluator.multiply(m1[i][v], m2[v][j],auxiliar)
                evaluator.add(auxiliar2,auxiliar)
            row.append(auxiliar2)
        result.append(row) # append the new row into the final result
    return np.asarray(result)

def mat_transpose(m): 
    for row in m : 
        rez = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))] 
    return np.asarray(rez)

def example_basics_i():

	encr, eva, dec, enc, con = initialize_encryption()
	
	# encrypt two matrices
	X = create_matrix(10,10,2)
	X_E = mat_encrypt(X, encr, enc)
	X_D = mat_decrypt(X_E, dec, enc)
	print("______Matrix 1_________")
	print(X_D)

	Y = create_matrix(10,10,3) 
	Y_E = mat_encrypt(Y, encr, enc)
	Y_D = mat_decrypt(Y_E, dec, enc)
	print("______Matrix 2_________")
	print(Y_D)
	
	s = scal_encrypt(4, encr, enc)
	X_2 = mat_mulscal(X_E, s, eva)
	X_2 = mat_decrypt(X_2, dec, enc)
	print("______Scalar multiplication_________")
	print(X_2)

	I = mat_id(50,encr,enc)
	I_2 = mat_decrypt(I, dec, enc)
	print("______Identity matrix_________")
	print(I_2)
	
	XmY = mat_sub(X_E, Y_E, eva, encr, enc)
	XmY = mat_decrypt(XmY, dec, enc)
	print("______Matrix substract_________")
	print(XmY)

	XpY = mat_add(X_E, Y_E, eva)
	XpY = mat_decrypt(XpY, dec, enc)
	print("______Matrix addition_________")
	print(XpY)
	 
	XpY = mat_mul(X_E, Y_E, eva,encr,enc)
	XpY = mat_decrypt(XpY, dec, enc)
	print("______Matrix Multiplication_________")
	print(XpY)

	Yt = mat_transpose(Y_E)
	Yt = mat_decrypt(Yt, dec, enc)
	print("______Matrix Transpose_________")
	print(Yt)
    
def main():
	example_basics_i()
	input('Press ENTER to exit')

def print_example_banner(title, ch='*', length=78):
	spaced_text = ' %s ' % title
	print(spaced_text.center(length, ch))

def print_parameters(con):
	print("/ Encryption parameters:")
	print("| poly_modulus: " + con.poly_modulus().to_string())
	# Print the size of the true (product) coefficient modulus
	print("| coeff_modulus_size: " + (str)(con.total_coeff_modulus().significant_bit_count()) + " bits")
	print("| plain_modulus: " + (str)(con.plain_modulus().value()))
	print("| noise_standard_deviation: " + (str)(con.noise_standard_deviation()))

if __name__ == '__main__':
	main() 
