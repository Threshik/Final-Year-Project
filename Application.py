from google.colab import drive
drive.mount('/content/drive')
!pip install gradio
!pip install fpdf
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gradio as gr
import folium
import fpdf
import os

# Load trained model
model = load_model("/content/drive/MyDrive/Indian-monuments/InceptionV3_model.h5")

# Class labels
class_labels = ['Ajanta Caves', 'Charar-E- Sharif', 'Chhota_Imambara', 'Ellora Caves', 'Fatehpur Sikri',
                'Gateway of India', 'Humayun_s Tomb', 'India gate pics', 'Khajuraho', 'Sun Temple Konark',
                'alai_darwaza', 'alai_minar', 'basilica_of_bom_jesus', 'charminar', 'golden temple',
                'hawa mahal pics', 'iron_pillar', 'jamali_kamali_tomb', 'lotus_temple', 'mysore_palace',
                'qutub_minar', 'tajmahal', 'tanjavur temple', 'victoria memorial']

# Monument metadata
monument_info = {
    'Ajanta Caves': {
        'history': """The Ajanta Caves, located in Maharashtra, India, are rock-cut Buddhist cave monuments that date back to the 2nd century BCE. These caves are renowned for their intricate sculptures and murals, which depict the life of the Buddha and Buddhist teachings. The Ajanta Caves were abandoned around the 6th century CE and rediscovered in 1819 by a British officer. The caves are now a UNESCO World Heritage site and a symbol of India's rich cultural heritage.""",
        'preservation': 'Well-preserved',
        'location': 'Maharashtra, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Ajanta+Caves',
        'coords': (20.5305, 75.7007),
    },
    'Charar-E Sharif': {
        'history': """Charar-E-Sharif is a revered shrine in Kashmir, India, dedicated to the Sufi saint Sheikh Noor-ud-Din Noorani (also known as Nund Rishi). The shrine, located in the town of Charar-E-Sharif, was destroyed in a fire during the insurgency of the 1990s but has been rebuilt. It is a significant place of pilgrimage for Muslims and attracts visitors for its spiritual atmosphere and historical significance.""",
        'preservation': 'Rebuilt after fire damage',
        'location': 'Kashmir, India',
        'damage': 'Fire damage during the 1990s, but rebuilt',
        'map': 'https://www.google.com/maps/place/Charar+e+Sharif',
        'coords': [34.0847, 74.6431],
    },
    'Chhota Imambara': {
        'history': """The Chhota Imambara, located in Lucknow, Uttar Pradesh, was built in 1838 by Nawab Muhammad Ali Shah. This beautiful monument is known for its impressive Islamic architecture, combining Persian and Mughal styles. The Chhota Imambara is also home to the tombs of the Nawab and his family. The monument is famous for its intricate chandeliers and decorative elements.""",
        'preservation': 'Good condition, undergoing periodic restoration',
        'location': 'Lucknow, Uttar Pradesh, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Chhota+Imambara',
        'coords': [26.8478, 80.9488],
    },
    'Ellora Caves': {
        'history': """The Ellora Caves, located in Maharashtra, India, are a complex of 34 rock-cut temples and monasteries that date back to the 6th to 10th century CE. The caves feature a blend of Buddhist, Hindu, and Jain temples and are famous for their large-scale sculptures and intricate carvings, particularly the Kailasa temple, which is a single monolithic structure. The Ellora Caves are also a UNESCO World Heritage site.""",
        'preservation': 'Well-preserved',
        'location': 'Maharashtra, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Ellora+Caves',
        'coords':[20.0294, 75.2173]
    },
    'Fatehpur Sikri': {
        'history': """Fatehpur Sikri, located in Uttar Pradesh, India, was the capital of the Mughal Empire under Emperor Akbar in the late 16th century. The city was constructed in red sandstone and is a blend of Persian, Mughal, and Indian architectural styles. The city was abandoned shortly after Akbar's reign, possibly due to water scarcity. It remains a UNESCO World Heritage site and an important example of Mughal architecture.""",
        'preservation': 'Good condition',
        'location': 'Uttar Pradesh, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Fatehpur+Sikri',
        'coords':[27.0948, 77.6611],
    },
    'Gateway of India': {
        'history': """The Gateway of India is an iconic archway located in Mumbai, India, overlooking the Arabian Sea. Built-in 1924 to commemorate the visit of King George V and Queen Mary to India, the monument is a symbol of British colonial rule in India. It also became the site of the departure of the last British troops from India in 1948, marking the end of British rule.""",
        'preservation': 'Good condition',
        'location': 'Mumbai, Maharashtra, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Gateway+of+India',
        'coords':[18.9219, 72.8347],
    },
    'Humayun_s Tomb': {
        'history': """Humayun's Tomb, located in Delhi, India, is the tomb of Mughal Emperor Humayun. Built in 1570, the tomb is a stunning example of Mughal architecture and was the first garden tomb in India. Its design inspired later Mughal tombs, including the Taj Mahal. The tomb is surrounded by beautiful gardens and was designated a UNESCO World Heritage site in 1993.""",
        'preservation': 'Good condition',
        'location': 'Delhi, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Humayun+Tomb',
        'coords':[28.5930, 77.2502],
    },
    'India gate pics': {
        'history': """India Gate is a war memorial located in the heart of New Delhi, India. Built in 1931, it commemorates the soldiers of the British Indian Army who died in World War I and other conflicts. The monument stands at 42 meters tall and is an important national symbol. Every year, the Republic Day parade takes place near India Gate.""",
        'preservation': 'Well-preserved',
        'location': 'New Delhi, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/India+Gate',
        'coords':[28.6129, 77.2295],
    },
    'Khajuraho': {
        'history': """The Khajuraho Group of Monuments, located in Madhya Pradesh, India, is a UNESCO World Heritage site. The temples, built between the 9th and 11th centuries, are known for their intricate and sensual carvings, depicting various aspects of life, including deities, animals, and human figures. The temples are an important example of medieval Indian architecture and sculpture.""",
        'preservation': 'Good condition',
        'location': 'Madhya Pradesh, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Khajuraho',
        'coords':  [24.8497, 79.9192],
    },
    'Sun Temple Konark': {
        'history': """The Sun Temple at Konark, located in Odisha, India, was built in the 13th century by King Narasimhadeva I of the Eastern Ganga Dynasty. The temple is dedicated to the Sun God and is designed as a massive chariot with twelve wheels and seven horses. It is a UNESCO World Heritage site and is renowned for its architectural brilliance and intricate carvings.""",
        'preservation': 'Well-preserved, though some parts are damaged',
        'location': 'Konark, Odisha, India',
        'damage': 'Some parts damaged due to time and weathering',
        'map': 'https://www.google.com/maps/place/Sun+Temple+Konark',
        'coords':[19.8883, 86.0922],
    },
    'alai_darwaza': {
        'history': """The Alai Darwaza is a monumental gateway located in Delhi, India, and is part of the Qutb Complex. Built in 1311 by Sultan Ala-ud-Din Khilji, the gateway is an excellent example of Indo-Islamic architecture. The gateway is known for its stunning arches, intricate carvings, and use of red sandstone.""",
        'preservation': 'Good condition',
        'location': 'Delhi, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Alai+Darwaza',
        'coords': [28.5245, 77.1857],
    },
    'alai_minar': {
        'history': """The Alai Minar, located in the Qutb Complex in Delhi, India, was started by Sultan Ala-ud-Din Khilji in the early 14th century but was left incomplete after his death. The minaret was intended to be twice the height of the Qutb Minar, but it remains unfinished. It stands as a testament to the architectural ambitions of the Sultan.""",
        'preservation': 'Incomplete structure, partially damaged',
        'location': 'Delhi, India',
        'damage': 'Unfinished and partially damaged',
        'map': 'https://www.google.com/maps/place/Alai+Minar',
        'coords':[28.5243, 77.1852],
    },
    'basilica_of_bom_jesus': {
        'history': """The Basilica of Bom Jesus, located in Old Goa, India, is a UNESCO World Heritage site and one of the oldest churches in India. Built in 1605, it is dedicated to Saint Francis Xavier, whose mortal remains are preserved in a silver casket inside the church. The church is an important example of Baroque architecture and is a significant pilgrimage site for Christians in India.""",
        'preservation': 'Well-preserved',
        'location': 'Old Goa, Goa, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Basilica+of+Bom+Jesus',
        'coords': [15.4909, 73.8187],
    },
    'charminar': {
        'history': """The Charminar, located in Hyderabad, India, was built in 1591 by Sultan Muhammad Quli Qutb Shah. The monument is a mosque and a symbol of the city's rich history. It has four grand arches and is situated in the heart of the old city of Hyderabad. The Charminar is also a popular tourist attraction and is known for its vibrant markets and surrounding historical sites.""",
        'preservation': 'Good condition',
        'location': 'Hyderabad, Telangana, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Charminar',
        'coords':[17.3616, 78.4747],
    },
    'golden temple': {
        'history': """The Golden Temple, also known as Harmandir Sahib, is located in Amritsar, Punjab, India. It is the holiest Gurdwara in Sikhism and was founded by Guru Ram Das in 1581. The temple is renowned for its stunning architecture, with its golden dome and serene surroundings. It is a major pilgrimage site for Sikhs and attracts millions of visitors annually.""",
        'preservation': 'Well-preserved',
        'location': 'Amritsar, Punjab, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Golden+Temple',
        'coords':[31.6200, 74.8765],
    },
    'India Gate pics': {
        'history': """India Gate is a war memorial located in New Delhi, India, built to honor the soldiers who died during World War I and other conflicts. The monument is a key symbol of India’s rich heritage and the sacrifices made during colonial times.""",
        'preservation': 'Well-preserved',
        'location': 'New Delhi, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/India+Gate',
        'coords':[28.6129, 77.2295],
    },
    'iron_pillar': {
        'history': """The Iron Pillar of Delhi is an ancient structure located in the Qutb Complex, Delhi, India. It was constructed by Chandragupta II in the 4th century CE. The pillar is notable for its rust-resistant composition, made from a unique blend of metals. It stands at a height of 7.21 meters and is a marvel of ancient Indian metallurgy. The pillar has inscriptions that provide insights into the history and culture of the Gupta Empire.""",
        'preservation': 'Well-preserved',
        'location': 'Delhi, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Iron+Pillar+of+Delhi',
        'coords':[28.5246, 77.1853],
    },
    'jamali_kamali_tomb': {
        'history': """The Jamali Kamali Tomb is located within the Mehrauli Archaeological Park, Delhi, India. This tomb complex houses the graves of two Sufi saints, Jamali and Kamali. The tomb is famous for its beautiful Mughal architecture, featuring decorative arches and inscriptions. It is a peaceful place and a significant example of medieval Islamic architecture in India.""",
        'preservation': 'Good condition',
        'location': 'Delhi, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Jamali+Kamali+Tomb',
        'coords':[28.5455, 77.2004],
    },
    'lotus_temple': {
        'history': """The Lotus Temple, located in New Delhi, India, is a Baháʼí House of Worship that opened to the public in 1986. The temple is designed in the shape of a lotus flower, symbolizing purity and beauty. The building is made of white marble, and the structure is a modern architectural wonder. The Lotus Temple is known for its serene environment and is open to people of all faiths for prayer and meditation.""",
        'preservation': 'Well-preserved',
        'location': 'New Delhi, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Lotus+Temple',
        'coords':[28.5535, 77.2588],
    },
    'mysore_palace': {
        'history': """The Mysore Palace, located in Mysore, Karnataka, India, is the official residence of the Wadiyar dynasty. The palace was originally built in the 14th century, but the current structure, built in 1912, is a mix of Hindu, Muslim, Rajput, and Gothic architectural styles. The palace is famous for its grand design, expansive courtyards, and vibrant lighting during the annual Dasara festival.""",
        'preservation': 'Well-preserved',
        'location': 'Mysore, Karnataka, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Mysore+Palace',
        'coords':[12.3050, 76.6550],
    },
    'qutub_minar': {
        'history': """The Qutub Minar, located in Delhi, India, is a towering minaret built in 1193 by Qutb-ud-Din Aibak, marking the beginning of Muslim rule in India. Standing at 72.5 meters, it is the tallest brick minaret in the world and is a UNESCO World Heritage site. The Qutub Minar is intricately carved with inscriptions in Arabic, depicting the history of the time.""",
        'preservation': 'Good condition',
        'location': 'Delhi, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Qutub+Minar',
        'coords':(28.5244, 77.1855),
    },
    'tajmahal': {
        'history': """The Taj Mahal, located in Agra, Uttar Pradesh, India, is one of the most iconic monuments in the world. Built by Mughal Emperor Shah Jahan in memory of his wife Mumtaz Mahal, the Taj Mahal is a symbol of love and an architectural masterpiece. The white marble mausoleum is adorned with intricate carvings and is surrounded by beautiful gardens and water features. It is a UNESCO World Heritage site and a major tourist attraction.""",
        'preservation': 'Well-preserved, undergoing some restoration work',
        'location': 'Agra, Uttar Pradesh, India',
        'damage': 'None detected, some restoration work underway',
        'map': 'https://www.google.com/maps/place/Taj+Mahal',
        "status": "Well-preserved under ASI",
        "coords": (27.1751, 78.0421),
    },
    'tanjavur temple': {
        'history': """The Brihadeeswarar Temple, also known as the Tanjavur Temple, is located in Thanjavur, Tamil Nadu, India. Built by the Chola king Raja Raja Chola I in the 11th century, this temple is one of the largest in India. It is dedicated to Lord Shiva and is a fine example of Dravidian architecture. The temple is known for its massive dome, which is one of the largest in the world, and its intricately carved sculptures and inscriptions.""",
        'preservation': 'Well-preserved',
        'location': 'Thanjavur, Tamil Nadu, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Brihadeeswarar+Temple',
        'coords':(10.7840, 79.1315),
    },
    'victoria memorial': {
        'history': """The Victoria Memorial is a large marble building located in Kolkata, India. It was built in memory of Queen Victoria, the longest-reigning monarch of Britain, after her death in 1901. The memorial was designed by Sir William Emerson and combines British and Mughal architectural styles. Today, the Victoria Memorial serves as a museum showcasing artifacts from the British colonial era and is a major tourist destination in Kolkata.""",
        'preservation': 'Well-preserved',
        'location': 'Kolkata, West Bengal, India',
        'damage': 'None detected',
        'map': 'https://www.google.com/maps/place/Victoria+Memorial',
        'coords':(22.5440, 88.3426),
    }
}

def assess_damage(conf):
    if conf > 0.85:
        return "Low"
    elif conf > 0.6:
        return "Moderate"
    else:
        return "High"

def generate_map(name):
    info = monument_info.get(name.lower())
    if info and "coords" in info:
        lat, lon = info["coords"]
        m = folium.Map(location=[lat, lon], zoom_start=14)
        folium.Marker([lat, lon], popup=name).add_to(m)
        path = f"/content/{name}_map.html"
        m.save(path)
        return path
    return None

def generate_pdf(name, confidence, risk, image_path, map_file):
    info = monument_info.get(name.lower(), {})
    history = info.get("history", "History not available.")
    status = info.get("status", "Preservation status not available.")
    coords = info.get("coords", None)
    location = info.get("location", "Location not available.")
    damage = info.get("damage", "Damage not available.")
    map_link = info.get("map", "Map not available")

    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, f"Monument Classification Report", ln=True, align="C")
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Name: {name}", ln=True)
    pdf.cell(200, 10, f"Prediction Confidence: {confidence:.2f}", ln=True)
    pdf.cell(200, 10, f"Structural Damage Risk: {risk}", ln=True)
    pdf.multi_cell(0, 10, f"History: {history}")
    pdf.cell(200, 10, f"Preservation Status: {status}", ln=True)
    pdf.cell(200, 10, f"Location: {location}", ln=True)

    if image_path:
        try:
            pdf.image(image_path, x=10, y=None, w=100)
        except:
            pdf.cell(200, 10, "Image could not be embedded", ln=True)

    map_note = "Map included separately as HTML." if map_file else "Map not available"
    pdf.cell(200, 10, f"Map Location: {map_note}", ln=True)

    out_path = f"/content/{name}_report.pdf"
    pdf.output(out_path)
    return out_path

def predict(image):
    resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(resized) / 255.0, axis=0)
    predictions = model.predict(img_array)[0]
    idx = np.argmax(predictions)
    predicted_class = class_labels[idx]
    confidence = predictions[idx]
    risk = assess_damage(confidence)

    # Save input image
    input_path = f"/content/{predicted_class}_input.jpg"
    image.save(input_path)

    map_path = generate_map(predicted_class)
    pdf_path = generate_pdf(predicted_class, confidence, risk, input_path, map_path)

    info = monument_info.get(predicted_class.lower(), {})
    history = info.get("history", "Not available.")
    status = info.get("preservation", "Not available.")
    coords = info.get("coords", "Not available.")
    location = info.get("location", "Not available.")
    damage = info.get("damage", "Not available.")
    map_link = info.get("map", "Not available")

    details = f"""
    <b>Name:</b> {predicted_class}<br>
    <b>Prediction Confidence:</b> {confidence:.2f}<br>
    <b>Structural Damage Risk:</b> {risk}<br>
    <b>Preservation Status:</b> {status}<br>
    <b>Location:</b> {location}<br>
    <b>History:</b> {history}<br>
    <b>Structural Damage:</b> {damage}<br>
    <b>Map:</b> {'<a href="file/' + map_path + '" target="_blank">View Map</a>' if map_path else 'Not Available'}
    """

    return {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}, details, pdf_path

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5),
        gr.HTML(label="Details"),
        gr.File(label="Download Report")
    ],
    title="🕌 Indian Monument Classifier with Report",
    description="Upload an image of a monument. Get prediction, damage risk, map, preservation status, and PDF report."
)

interface.launch(share=True)
