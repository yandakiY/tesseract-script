import easyocr

reader = easyocr.Reader(['fr']) # this needs to run only once to load the model into memory
result = reader.readtext('facture_test15.png' , detail=0 , paragraph=True)

print("result",result)