import qrcode

data = "https://anicos-invite.netlify.app/"
qr = qrcode.make(data)
qr.save("qrcode.png")
print("QR code generated and saved as 'qrcode.png'")