from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io, base64, os

app = Flask(__name__)

# ── Model ─────────────────────────────────────────────────────────────────────
class WheatNet(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        base = models.mobilenet_v3_large(weights=None)
        in_f = base.classifier[0].in_features
        base.classifier = nn.Sequential(
            nn.Linear(in_f, 512), nn.Hardswish(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.Hardswish(), nn.Dropout(dropout * 0.6),
            nn.Linear(256, num_classes),
        )
        self.model = base

    def forward(self, x):
        return self.model(x)

CLASS_NAMES = ["Black Rust", "Brown Rust", "Healthy Wheat", "Yellow Rust"]

CLASS_INFO = {
    "Black Rust": {
        "color": "#c0392b",
        "emoji": "🔴",
        "desc": "Caused by Puccinia graminis. Dark brown-black pustules on stems and leaves.",
        "severity": "High",
        "recommendations": [
            {
                "category": "Immediate Action",
                "icon": "🚨",
                "steps": [
                    "Isolate affected plants immediately to prevent spread",
                    "Remove and destroy heavily infected plant debris",
                    "Apply fungicide within 24–48 hours of detection",
                    "Notify nearby farmers about the outbreak"
                ]
            },
            {
                "category": "Fungicide Treatment",
                "icon": "💊",
                "steps": [
                    "Apply Propiconazole (Tilt 25 EC) at 1 ml/L water",
                    "Alternatively use Tebuconazole 250 EC at 1 ml/L",
                    "Repeat spray every 10–14 days if conditions persist",
                    "Spray during early morning or late evening"
                ]
            },
            {
                "category": "Cultural Practices",
                "icon": "🌾",
                "steps": [
                    "Avoid overhead irrigation — use drip or furrow",
                    "Improve field drainage to reduce humidity",
                    "Plant resistant varieties: PBW-343, HD-2781",
                    "Maintain crop spacing for adequate airflow"
                ]
            },
            {
                "category": "Prevention",
                "icon": "🛡️",
                "steps": [
                    "Seed treatment with Carboxin 37.5% + Thiram 37.5%",
                    "Rotate crops — avoid continuous wheat planting",
                    "Monitor field weekly during flowering stage",
                    "Remove volunteer wheat plants from field borders"
                ]
            }
        ]
    },
    "Brown Rust": {
        "color": "#e67e22",
        "emoji": "🟠",
        "desc": "Caused by Puccinia triticina. Orange-brown pustules scattered on leaves.",
        "severity": "Medium",
        "recommendations": [
            {
                "category": "Immediate Action",
                "icon": "⚠️",
                "steps": [
                    "Scout field thoroughly to assess disease spread",
                    "Mark infected zones for targeted treatment",
                    "Begin fungicide application if >5% leaf area affected",
                    "Document infection date for tracking"
                ]
            },
            {
                "category": "Fungicide Treatment",
                "icon": "💊",
                "steps": [
                    "Apply Mancozeb 75 WP at 2 g/L water",
                    "Propiconazole 25 EC at 1 ml/L for systemic control",
                    "Use Azoxystrobin for protective coverage",
                    "Ensure thorough leaf coverage during spraying"
                ]
            },
            {
                "category": "Cultural Practices",
                "icon": "🌾",
                "steps": [
                    "Avoid late sowing — sow at recommended dates",
                    "Apply balanced NPK fertilizer (avoid excess nitrogen)",
                    "Plant resistant varieties: WH-147, PBW-550",
                    "Burn crop residues after harvest"
                ]
            },
            {
                "category": "Monitoring",
                "icon": "🔍",
                "steps": [
                    "Check lower leaves weekly for orange pustules",
                    "Use disease forecast tools if available",
                    "Track temperature and humidity — rust favours 15–22°C",
                    "Re-evaluate 7 days after treatment for regrowth"
                ]
            }
        ]
    },
    "Healthy Wheat": {
        "color": "#27ae60",
        "emoji": "🟢",
        "desc": "No disease detected. Crop appears healthy and normal.",
        "severity": "None",
        "recommendations": [
            {
                "category": "Maintenance",
                "icon": "✅",
                "steps": [
                    "Continue current agronomic practices",
                    "Maintain regular irrigation schedule",
                    "Apply top-dress nitrogen at tillering stage",
                    "Keep field records for future reference"
                ]
            },
            {
                "category": "Preventive Care",
                "icon": "🛡️",
                "steps": [
                    "Scout field every 7–10 days during growing season",
                    "Apply preventive fungicide at booting stage",
                    "Ensure proper weed management",
                    "Check for insect pests alongside disease scouting"
                ]
            },
            {
                "category": "Optimisation",
                "icon": "📈",
                "steps": [
                    "Conduct soil testing to balance nutrient levels",
                    "Ensure adequate micronutrient supply (Zinc, Boron)",
                    "Optimize plant population for maximum yield",
                    "Plan harvest schedule based on maturity index"
                ]
            },
            {
                "category": "Upcoming Watch Points",
                "icon": "👁️",
                "steps": [
                    "Monitor closely during humid or rainy spells",
                    "Watch for early signs at crop canopy edges",
                    "Keep fungicides available as a precaution",
                    "Compare weekly photos to detect subtle changes"
                ]
            }
        ]
    },
    "Yellow Rust": {
        "color": "#d4ac0d",
        "emoji": "🟡",
        "desc": "Caused by Puccinia striiformis. Yellow-orange stripes along leaf veins.",
        "severity": "High",
        "recommendations": [
            {
                "category": "Immediate Action",
                "icon": "🚨",
                "steps": [
                    "Apply fungicide immediately — yellow rust spreads fast",
                    "Treat entire field, not just symptomatic areas",
                    "Inform local agriculture department if widespread",
                    "Avoid field operations that spread spores"
                ]
            },
            {
                "category": "Fungicide Treatment",
                "icon": "💊",
                "steps": [
                    "Propiconazole 25 EC at 1 ml/L — most effective",
                    "Tebuconazole + Trifloxystrobin for systemic action",
                    "Apply with high-volume sprayer for canopy penetration",
                    "Two sprays: at first symptom and 14 days later"
                ]
            },
            {
                "category": "Cultural Practices",
                "icon": "🌾",
                "steps": [
                    "Avoid dense planting — improves air circulation",
                    "Reduce nitrogen topdress if disease is severe",
                    "Plant resistant varieties: K-307, HD-3086, PBW-771",
                    "Early sowing reduces yellow rust risk significantly"
                ]
            },
            {
                "category": "Long-term Strategy",
                "icon": "📋",
                "steps": [
                    "Adopt integrated disease management (IDM) practices",
                    "Use weather-based advisory services",
                    "Participate in regional disease surveillance programs",
                    "Switch to rust-resistant cultivars in next season"
                ]
            }
        ]
    }
}

WHEAT_PHOTOS = [
    {
        "url": "https://images.unsplash.com/photo-1437252611977-07f74518abd7?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8d2hlYXQlMjBoYXJ2ZXN0fGVufDB8fDB8fHww",
        "title": "Wheat harvest season",
        "tag": "Harvest"
    },
    {
        "url": "https://images.presentationgo.com/2025/04/green-wheat-field-sunrise.jpg",
        "title": "Green wheat field at sunrise",
        "tag": "Field"
    },
    
        
    {
        "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7h2ZfTLJCOPaaJ-deNmrKlm0Q0gpYHp1aHg&s",
        "title": "Healthy wheat leaves",
        "tag": "Healthy"
    },
    {
        "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFlw5xFjf3AnMGcFnwxuODykJqwCth4PWDAA&s",
        "title": "Wheat field overview",
        "tag": "Field"
    },
    {
        "url": "https://www.adama.com/uk/sites/adama_uk/files/styles/hero_desktop/public/2021-12/Seporia%20Hero%20Banners_Wheat_920x665_0.jpg?h=e7aa7c52&itok=uLBKZ-Az",
        "title": "Wheat disease — Septoria",
        "tag": "Disease"
    },
    {
        "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRATeGIdTh3Haq42EdWljbCOzANYiRzTaRzBA&s",
        "title": "Rust infected wheat leaf",
        "tag": "Disease"
    },
    {
        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMTEhUTExMVFhUVFxcXFRcXFRUVGBUYFRcWFxUVFRcYHSggGBolHRcVITEhJSkrLi4uGB8zODMtNygtLisBCgoKDg0OGxAQGy4lICUtLS0vLS0tLS0tLS0tLS0tNS0tLS0tLS0tLy0tLS0tLy0tLS0tLS0tLS0tLS0tLS0tLf/AABEIALcBEwMBIgACEQEDEQH/xAAcAAABBQEBAQAAAAAAAAAAAAAFAAIDBAYBBwj/xAA/EAABAwIEAwYDBgUDAwUAAAABAAIDBBEFEiExQVFhBhMicYGRMqGxFCNCUsHwBzNygtFi4fFTkqIVJGPC0v/EABoBAAMBAQEBAAAAAAAAAAAAAAIDBAEABQb/xAAuEQACAgICAgIBAgUEAwAAAAAAAQIRAxIhMQRBEyJRMmEUI3HR8AWhseFCUpH/2gAMAwEAAhEDEQA/AA8GFfbHRsfL3bY7mR4JJ0u4tiZtp5cXE6Wvdxqlw8RGWninbJcNY5zHCNzW3DnE3LSSfzOzXB0ACL9huy9PNEZ5o5J3sc5rYmENYQA0kE3AJvwLrabFTfxGxGcUjIX0sdNE5/gYJGudZgvYtYMrRqNivMgvpYU+Ezz3DBd5K9B7EdpXwB0bYnytPjyxtLnDcOOmoFg3fTyXnVC+wJW6wzDJ3QMdTwvilGomE7Wl5aPhyX2PI8ddEltxybIHx1Zc7b4xT1MEUtK+5LjJIzUWIFrPZ+a5d89bm5C4PXuJ1Dcmt/E4i54tNvkdUOr8SkfUHvWOZKBllAY1pdbd0jLak3uXDffRE8PgDW2HMk+ZUvm5LtyRTG4S4DOK4k6d+Z2gAAa3g0AcFo/4fUQ+8nI/0N9PE7/6+yxwaSbBek9nGdxRMLxazTI4cRmu63nbRB/psXkzvJL1yZN8FLtziYZGIr6v1dbfIOHqfoViYsecKV8DAGtedXDUuzHxAnh4QVcw+KfEaszXDWtOoLSWtbsGjg51rnzN0O7bUjaR/dMe4gNL/Fbd5IGwA4OVmfecnOPXQFqMeQLh/ikceth5DX6r0vsji7GM7nKc7fH8TBmD/ECLuueS867O0pdlYN3kNHm82B91q+3mHyNewmKNrQMolj8JOwDXg7W4apGJuE5ZI+qR2JXGmehSVR7tzixzbAkXym+n+kleS187pngfhYMreW+Z7v7nX9LI47Fpo6JkLmFpkbaOUODmuafiJ5OAv8kDpWaabcAs87yXNRS/ASjTo2PYShu58pGjBkb5nVx9sv8A3Fa+ofZpPRVcGou5gZHxAu7+p2rvmfkou0lRkppXcmG3nbRepgx/B49ftbFt7SHdn5c0DHfmF/fVPxutMURLdXusyMc3vOVvzKZ2fjy08Q5MaPkh8j+/rg0fBTDM7rI8WaPRtz6pmzWNJds6vsGsPphHG1g/CAL8zxJ6k3KsJrioayoDI3POzQT7C6c2or+gNWwHRu76vkd+GBgYP6n6u+QstFdAOxcBEBkd8UznSH1OnyR+yT4qemz98hZO6O3SamPPBCO0mNinZlbrI74Ry/1H96o8mWOOLlLpGRi26R3GMRcXimhP3j/id/028XefJEqOmbExrGCwaPfmT1O6F9mMLMTDJJrLJ4nk7i+wRlLwKUv5k+30vwv87Cm0vqhwK7dNSuqRZ0lK64kFxw665dcSXHHbpXTV1ccPSXElpx5Z2cxl7IjTUpabuuZ3WDIwRrkb+Ii1h9OKCfxUrw98EbXlwjiAu43Jc4+Iu62DfdB6ftSyOMhzHOB3LTbLrbjw2QbGqoPeSL2sLX3tbivHwfJ1LoLyJR1qJzDmlxawbvc0DzcbLatjq6V+UPBO9tbAf4WV7KgfaoidmuzH+0XWz7UYkJHZmHduU9P3dDmX2pdg4OIWZyCR0sr5Xm7nuJ9yj0YsEJpWgW6IrGCduS8vypbyG4k0NdPZ7WC13C4vxy/hHC56ohW46+ajihjvG0OMcuri4uO1r7j4jYdOawprD3oL75A83cOTtvnZazAKwRzhoDXR3b3ch1yvN/iJ5gDbkq8eN4o8e1yUfqWp6ZglC+CmZHHY+H4n6anXUDUryDtvWOkrJA5wcWuEdxoPB4TYeeZeu1GINZG90kl8jC428IsBc29l4Qx5dJmdqTdx8ydfqVe5xcVXSJPItcG67A0mapj5Mu8/2jT55Vt+0lP37m097ZwSbi402uFkuzEjIbPfm1AAIJBvx234KziWPGESStcS4tLIs2pDn7G/TU+ighmi4afmXP8AQfCOi2M7I0tcYM+dsDnRstsLuu/L0J0/tRGkBZZ4bcRZZHX2s1w3+voVn8Id94L/APJReLEHNlMIIaJWZQ463J4EcrWPukTg/ls7C9+z0+hxNk0bZIzcO9/UIP2sqi6PubfGQL+t0G7MzmPwsLQWkteOBLeI5c1Iaovlbn/6gPkL/wCFXm8xyhr+R3wqPJrKirbBTlx/C3QczsB7qt2QpS2HvH/HM4yOP9W3y+qz+OV32iVsTT4AQOhJNrrcRgNaANgLD0VnjZVlyNrqKpE840inilTkYTyWa7T41njEDPilIB6Anb1U2PVbs5Z+FZ/Bm97WM5NN/wDt1HzsvPy+XPJmcIvjoa4KMV+T0iigEcbWDZrQB6CymaU17tFBUVTWMLnGwAXuucYL9kifVsAY/wBojFJkYLkDU8BfgqfZjDnVEpqZtQD4b8SOI6BDaamNVORsHEueeTeXnwXoMMbWNDWiwAsB0C8bxIy8rI8uR/VPhFGR6Kl2TBJRulAC5DIDqvb3jdEurqyUhcSzLqIGjiSVkrLjhLhXdErhdaNpnF2yVk5aYJdSSWnHzgYgQ2wLbgklrtNL6cQdOaz9S67vVEmtYItRd27CCQWnYg8ChDzqvNwRpMXnldBbAaiSNxkY1rsos4Ovs7e1uK0+QSEvAsDrbe3qg3ZenLmuIcRdwBbpYjr1C00oDG2C83zc1T1XZX4+P+Xcuii1utlblqXRNzttoRfysSfoFBDoCT5+yo4lV/cvAOtruHQ2/QD3SMWPeXP9P/p0fbAtPU9/LIzKA1wJIHC53F+V0XwZxEYiIF7kknZ4ALsgP5hluFn+ztOXPLtRcHL+ivOidma+N9mg5xxtbf1JB9CvSyqOzigvHm62fZssexcSUmYHV4Y0jo4E3HQhrh5grH0I3PMq1iFX92IbghmZzSPxBxFr8rWcFXpBYD96oIQ0g0TeRLbIbyNmeFthqBcg6b7fKyzmLVBe8MGzNPNx+I+m3ui09V3UFwdToOp2aP3yQSkj18uPU7lS4oKEbG5ZXSJaOLxAfPyVmrg7wteBdwuRw2BuR7KRlMdDbQ6nlYaaet/YKzUUpkZaIE9dBbnqdEORu0x/iquyfs68xkyuIDZnBgvu02vm8jsiGIHLcEIfFTytgDe7ddl3OJAsSD8Ivv6IjjUoyj/UAfcJTtr7FOZquCvgj/vAeRutbLjlgbhZLs8dXaeqt10wzWulxy5INqDFQS0tktfKXNc8ql2UlySZz1TsQf8AdWauYPHYLLeOOy7sx/aaNXU44LaXA4rPYliLpbD8I2HM8yoKubccBuhjcVaH+EXA4rJ582X9TDpXSNJ2fqBCTfd36I67Fmkb2WQosQL3bNt80RmOmiLH5OXFDVDHjXbJ8RxR0l2t0B0v04qwMeEbbWOgQ6njTpqYOSoeVlUnK+TVFNU0X6DtIXk3bYcCr7Mej/EbehWeZEGCybJDdOh/qOaL74N+HHL0H5+0kYOlz5KF/amMcHH0WZNIblU6xhbsqv4+cnSYcfGxmoqe01/gHuuQ4u8jWyx7ZyFfpqxTZcmV87MpjgxpVRs6TFHDfVW3YzyaFkoa1TOq7broef5EFqmJn4UG7oOOxd990lmzXDmkg/jPJ/8AZhfweP8AB5DV+Ftid7enQqlDEXODRuVbxSs7zKLAZdPPkSp8CpC6S69/b48WzPm2t8lI1GCUzYo79NSgOO4/4i0GyL49ViOLKDwWCp4s0gLhcXuVB4mFZHLLMszzqKxxNj2fxS7XBwvm2ujVTDCGFz4tHWDj5IBS1DASWsOg0C0OC1Ze7u5GWaeJGmqHJFqVx4QWJKqI5qenaBJHcHQAcDyAQbEXATFxFw8DS2gsCL222uUYx7DJKdwc1mdlyQ5uuW+1he3qUCqa94DyYCA9oFy8GxF7O04+JHjhKw3KMQdVvaZDkFm3sB02+ep9URid4h7+2yCQHxBaDBqtrBI8tzOyhsbbNILidiDwvl221VeZUjzMf2lyW3yukIY27hHfbi4jUnoBp6lSfbWxEMDQ+Qalvp7acb2CGR1gijDC+z3gucQRe7tr6Wyi9+pT6WvjYPADIXHdxtfq43uRpfgpXjK4q3ZoIamWXixtuficflYH0IUVTA8kkyzFzRYNa62m/ha0W/VKmmeQ57YoLZbkNe4lp5ktFtuAVQ483OLx5CRYuDnON7bgXAAPnog74Q9xS5kWhFLGBI2dxaW5vFa4tbQG3y4o9/6s8iPI5zXSjwiQgh/GzXWshtPiQm8DWF+niblAvpYljiTbnbe/RTYgIxRtaM1o3AAPaLkO0PQ6G/BC4r3/AGDb4epWdVTQktf4C4k201B4ttoR5LktawjwgvcFFT17GNDJmh8JIAc/doO7mv3abAacco6qafAzH4xICx3wubsfMcCgeNHRfoloa9rnWIseIKJVFYIxw15LFVFQ5pJAJtxQ44wS9oF739ES8fblAyy0jUVFWXuyZrX3V9rI2tFmgdeaZh1Ex2UuAuUekpGEBgA0Uco+inGklZTrKSPu7jR3AjRLD5fBlPxjY33VpsbWuaHbBMqImtks0C2hB6cQlxdcD4pSXAShcHMBHDdJkoUH2kBzSz4HaG3PqquKT5XW/ZQyiq2AmqLGfM5W2NQ6inCvGS6Uovs5KjpamSUYOpCfG9P71cotBJsD1uF3GiCTQOYeK2twVWlpWlOhNrhjYzAGGucTqis0RLVMylaFYNrKiOTGlyjpTd8AB0RSRssaks+XEF8jPD3tu+3In6rT4VEI2XO6AYczMcx46++quYpiGVtgvX8lOdY0fN4motyZTxmpMr7cAr2DTNiY4FgJdxIvZA45AC0k6ndFoH2cXA3HLgtnCoqHobjdS3skFQGkZAST/wAo8zGi5mTJ4ggsVEJNWyWINy0fotDh8TR4DY6XDuPqlzQceAthsj5qdzCdSPUeSxfaDNG3uiQdQLkDNYa6nfktlQxFjS4c7+yzfbfDnENqWg5SPF0OyzDDXIkdnm5Y+OzJRusSeSu1VU+NkYGXNlcRzu82ueoF7f1FDoXagHXW9vzW1y+u3qrNQHMcXvsXh2lzcAnXKBxsTqentdNK+SPEqtkcOHueSZXG4Iu2xL7cyPwjXRbKm+zxC92i4AszV50F720HDck6arOUFW7KXyG+a9gePM/p6K0yraAdCWnkbEDopc20nX4/BXicVyHYCGjOwyBoN2ltxe51BAOp525JlTNTzOyztLXZvDJG3JlH+tuoPoEIpZZGjvKaYEjeJ2hty6+iNxzSStDshcWNu+NzLi19S3UnTTTpdTuLi7f/AGOUlLhIrwU0sT2k3I4PYbtPk7geh9lpY5nPiLJjeJwNiRc33zC3wuBvpxtpfVAMMxWJri0XY0n4ACRY7kG5vx0IKMCnZkuJM7X2ax4uAWlwAcW75mki/S46rJ3+DcSiotplSqa6ICNzBKweIAHR7dQS09OXQIhS0jJKUMizRuPiY12xAva1t7A6gcNbXFiGq6sC2bwzRWAI/EOvXYh3Ec9Fao69rGNGbS+bTQxOABBaOX6ErkqSNTTbQAqJywOjk0dxBH7uOqH0eFH4z6L0OtpoKqIF99fFn3fC4kG42vGeXLjsgs33V4nAOLbZSNiCAWuHQgg+qKeTRVD2DDFcrn6LdFfu262KMwSXGa+oWcpR3jSDoRqjfZnxxvHFpNwpsi4t9FcEn0WJn5x4vdMootbXvw8k2ncHBzeSp0tVlefOyQo0uBlU7NJBTtYx7bjmL8L/AO90PxCmE0LZGbs0d6bp1TA57S8bWXMChNpY3OtfVBGN3YySuIPiRCiqOBVK4KlY/LqttJUYo2X55MqeyUEKGaUObdUo57JTfJtIJmWyX2hA6it1U1JUXWxVsxtFyaoKqSYgQrTrKrPTgqj44sGyH/1XqkqjqDVJD8ETrZio/AxZ+tqi93kiOJ1FmlBIzbVfQ4Yf+TPnWy2wEWPT5q1h9YWOvuOKpma4v+9UreE77E6Jmtrky/twaHJkIqI7lj7Cw/DfojeGRF4zAnisvg2JWIiOgf8AJHqOJ1Kb5szCRb1UeSLRbCV8GgocWyxFpGxV3ELS0skcViS2+VVKGnErC21i7ZcwmMxvA4tNieYKmdpX+B6/VT9o86gpz4r6Zb5r7gNBLgOuwTYKYOcXWsCdr3sOp4nrzWg7aYYWVN7eB/iBGxPH9PZCyPCQFc8ratezz5R0epWqqwvOU7N0aOQGybBNuFSLvEp4DumaJRpAbtysbJJY6FajD8Ze7IXvLXM2eCRcflNllHDVEYHeFKywUo0HDI4uzYYjhsUj2vjdlkcM5a4jJLxzMd+Fx4g6eS7S4gH3glBYWnw2bYtcL2Dh5kC51F/NC4Zu8pXAkZoDdv5sp3AKrV2LPcI3Ot3jNA/iW8Gu52/VRaOTp+itzUfsvYex6lLzHK0nKfu3A6ODmlx18x9E2hjzte2MgSxguvzykOaB13t5K3BWxyNJlaWudEHucDcjIHZZADpre1uPmboN2caRVSszi4+F41Bym+nR3hQpN3foY6jJNezRw4hYvy6EtaWEXAI+KRljyJNvIKLtTTC8NQ0/z22LdrFjW6jpYj9nSlhcght32viYzS+wdntcbkHJ6aItRO7w1MDtQ1zpImkWcCRmBaevi04g6cFj4TGStkeDAXyuGp2UdHUGGqLB8LrA/wCVTiqcrw4eiifVZ5S4bgj5KaDk3z0Owyi417NJXRCK5B3QaKVriddSu9oZ3ANJOh0QfDXEvXKFJszJm+6ibTBKrRzXaiyioZ7zjcG5B5EKOmtbTkqLMUET3A+Y80rl/ZFG6jxLoISMPeOB3BK5K7RUcQxVj3tdGLXAz+a4atKlGSAWaN0i39o8NlUgm1T49kKmqspKYo7Pg6bqmW6l+quUsgGqBsmLlfguqljVUJlLkIy1B4JjawjdRseuBoKNRXR2zLArQkq/2fquo9EbueSV82Z1uCrPdcgLhuSeQT42+K/Je6kkjwyw7gFK14D2gnS2o5+fRQtOqdG4jM7g5pA6G+n76Jb6Oh3ZDK8k347/AOFscPnJpA5+oBWNI0J9FqMGqmmlLCRo4ac0ryI3FDcDqRssJnb3QkB2GinwmQyOcTs8WuhEcAbZrDoQFoHNEbI2s3P7+q8x9Oj1O3GzPduatzMtO5uu7XdOKz1LxZ0RztzXNlewfjjGvss1h1Rq48LFOxK8aoiz85mCpWWcfNPjKjkkuT5rrDovRXRE+xzD4lekdYKtSR8VfoMPfUSiNn9zuDW8Sf8ACTJpcvpG020kMw/MbnW2w6nc+2nuEqt9zbkjMYjzvDP5cQyM62+J3Uk3N/JA5HXeSkRe03wMmq4Nn2dmY+EGWxETZISP/ikaC5x/pyi3m5ZvDZRG8G9wHC5Gl23125hT9mMR7ubKRcSNLCOpt/uPVNxjD+4mdGCS3QsJ4scLtv14HqCgcam4v2OcrxxmvRp6sts5zWgsMje7AtoSHE+Wuiv0GJONQ15A1tG/hexN9eDh4bHoUGwWImMtcT8JLePiBDwPYOHqFPSsfJK1jRrmPef3tzXHOwO3VTJctFzdx3AeLOdHMWflOh5tOx9lc7Puu91+Kn7WgGKKQtySNe+LLxLQS4j+1xt/eEFwwuzAhFKKeP8ABNjn8eX8mt7Qxjum89NELpqfIMxRCSIusXcFQxSp4BRRbf1Q/O1vsWaKuJvrohNbJd6rPqcuy5S3LrlUxx68iJZnKKiEoRYKdjlAX2TidErXZhwJ5cTAGVBhIXO9U6WPW6kpwE+OOONDJZG+wpQsAREgIK95FrK19oNkOrfJzmiy5cElkOknI3TBWApkItg7hU1BSQ9s6Sb8Z2zPMA7TzKmg2uoDsFM3ay9hnlskB0TgSGWOocQR0yjUfP5qOQ6JSkZW2PA+hJt+gQUcujoPh89V2GQjYpjikF1GG9w8OeyNwP72WiopHNLe8Ot9FnOy8gfCGj4m/VXn1rszcwXjz/U0ezjpQT/IP7fUndOMgP8AMGizdDIO5dzW97f0wnomys3i1PlxXmr3gRiyq8dbY0r9kOePx5WRA7rrXpl02MEkAC5JsANyTsAriX2GMJp3yuDGC7nHQfqeQHNbfEIGUFGWsN5pTlzcT+Zw5AA2HmE7shhAgaM1u8cLvP5R+QHlzP8AhZftVjH2idzh/LZ4I/6R+L+46+Vl5Dm/IzaR/SuX+5UorFDZ9vhfsQwyZYyBxVTYJ7LkAKOU+IBVwVNkz6LmDvLHiQbtII9EVrJxPUmQAhtmaciGjMByGbMfVCYBror0EgYLlKmudg1L66+jQUsoJcAbOyODBwuRbQc7XHsmYZK7vozntrldxNi02Py+iH4ZWWdn/fX5LuFyNOdxbcd43zawEEkdfFbRSqPZ6GHJaSNH2ti7+mB0Do5GmThdtnRh/qSP/FB8IomNNy4FG8RaTRzGLWN7WvG12BskYc3XUjKSf2FlGTZW2C2XMNReb65NjR19ZCRlDwEMjw1r7nvm39FnKh9zdSUpJ0RQwwxx4QmWRzdsKTYBIT4S13qrEOESRi7mlU6aYsOjjfzV445I0WzX80M5bcDIRVWQAeLVSylSx4yD8bAfJdDoHnR+U8ispR5GxdIHyMuutjRcYS8i7C1w6FU6qkkZ8THAc7ae675ITVJnO+2hkfVSF6HzVtlFBW3JT4YnrZPKfNBSdoIQ/uDfROa4uOivmwHVEjY8lS6STkk8M864kclZhbqq0eriURpGC56BWTdIga5KUzkraDy/3TJjqn236BacK6e43sob8FLEuMDnZ+rdG4ZTq46j6LZgiXKPxXWB7OvvO36LbtcBKLCztDZeZ5f1kej4kdo8sPsyxtfBLazmm1+K8ixNoacg4Ej04LdfxMmeBE7Ykb+mq83lfco/BxUtk+DPNnb1rlDmuWr7NYVkAlcPG7+WPyg/iPU/TzQ3sxgxmJkI8DP/ACPLyWwhIYDK/RrdT1ty/f1XeZ5CVwi+ff8AYlhB3Y3tDiPdU/d3+8k3PEM2Pvt78llsOoDM4NG27jyATcUrXTyXtq46DkOAR/uxTUriPiIsTzc7/CTD+RjUV+qX+f7Bupyv0gZTysBeeA0b5BDI/HJ5lclks23NOw1l3BUqOqchUpbcBuKksq8+uiIyShrEEY+5U+NuTbZskl0EohZuikwdv8wFxA8Gg3OtyemypSVGlgo6HMSbfi0P+F2jpjcU0pJm2ikdknaHDIIJMnJ2l3C39Nx5gFZwRHLc6Ba7Bc3cSCUWaWSWB/ECBm9rfRZetkMjtNGDYJEJIs8iFtSYNkGY6DRSxkMCkms1UHSJq+5K4/gm703uU9j8xsFRfLfZTQ1AYjeN0bF8/sX6iHI3qhBqbmyVdiDpNL6LlJBfVPw4nCNzCyzi3UQnh1VNH8D3DpfT2Wppu00wbZ7Wu8wsqZcoCmhrFPl8eGTlxAWaUOmXa6SOQ37vLzsqTqQA6X9VyeXkoDXpsMbjGkZLJs7fYTpPCrUjL8UIZU3XY67WyGpWYnwXSzqkoRIkh2kHsjAQbIlTusxx6Ic0KWR9hbmvTnHYjvkiA1HUqYA2cRz+i5MMrgOLW3Pmf2Ffwd7fgdyJ90M5VGxkYXLUFHdSjRvmm1bAH2HNKoPDkPqmd0A1zQW7ItBnBPDZaillc+dzhuwrF4VMY3By0nZuvvK6/FQeXCTbkizxpxTUX+Qj/EXFGysiZ+JpF1jcNwp08wjbtu535W8T58lpf4g0wMkRjFy7Sw4k7Il2cpRTstu82zHryHQJSz/D49x5bNzR3zc9BptAyNrYYxlAFv8Ac9VlO3eKjM2mjtljsX24ncN9Nz1I5I1jWK9xGXnVx0aObuH+fILz6iiMshc431zOJ4km59ypvAwNv5p+v+fyd5GRJaoP4BR2+8cNTowfqudq6wXbCDfL4nf1HgrzZcjDKdA0eEfRY2pnL3Fx3JJKrxQ+TK5vpEzdQr8j3yXRfC47C6DwsJIRdslgqMytaoV7L1Q/wnVD4Ba6YZyU+FhJsN0iMdEG+S1TxX8ytPhdEyBvey8NQFFgmFCMZ5N9/JB8exfvX5WnwjRS7vJPWPRVGPxR2l2a/CqovMT3kOicZGEW0vIAS2/kHLNVDwy7Qb2JAPMA7psWJf8AtmQ/lfmv6O//AF8kLqpDa62OJuVBZc21JHKqUlU8xOiUc3NStHGyt1UVSFfscAsuDVVKmcqONzimxxujLJ5pADZXaCpHFBZIjdSwghMeP60A+w9PKN1WbProogbixTBBbVJVJGSCL5RZVnQkqAX4q0yQtGyJLjgFEsEJA1XIdCSVKKklhJA/4VZxJCVG3djePRM+puUkNMhXE34kDyCGKxSszP6N1KgbtdXGDJC48XfrsPZNm6QuCt2UZZMznHmfkFyR1naeS5GNQPVNcdfVMS9HeyVhublMvcpx0aeuia7Ro5n6LkckTRPurFNOWOzDgqdIrlJBncRwG/lyS51Tsy6lZrMGnMxEsg+H4AfYu/RFYTmcSNr6fqUFw65OUaX08gmdqcT7lncsNnP0PRnH3291408Tnl0j7/2RZHI2tpAntLiX2iazNWN8LOvN3r9AFYoKewDR5uKoYfFYZjudkUFb3UTublfNaxUIL9iXbaVsr9pK69om7NGvUoJCdVyV99eJ3ShCox41CGqMbt2EoVHVS8F2NygdqUKXIPbLlBv1WswagDPE7Vx2HJDMGoAwB7tzsEQrazu23v4j8lBmk8ktYleKKgrZztViuVndsOp3KyVMbldqZi9xKlpI7KjHjWKFIVkm5uwpTtv5BNxGqYBbcqKeos3RZ2qkJPFbgxOTthJpRL4qgu/bL6IawFP7yyr0S6MsMRsZa5UEtQ0baoYZCdyrDXCyFYqdthPJxSQ105JUsOY+6hay5RKkcG7opulwKbCdBhLpSAE/FsJdDurWF4iYyH8FNiWLtqHWUkJOXo5UkZwnZWHzaWVp2H23Pko/sviF+abskjFbLFe3LA225Q6Bx4ovi8ws1o4BChYG5SfFl9LfthydOh5p0kvtzeRSVdnWgBDHcgcBqVzEJr2b+/3supLlzIBdEDD4ieX6aKJpSSTUYThly1vqVDUPu4nhsPRJJDHs1dE1INEapo8jbcTv1SSScot9hKlqhE0yHgCSsw6V1RM6R53Nz0HBo6JJJeCKqU/fQxyetBNmp8lRxSoubDYJJJkF9hcSgrES6knSCfQ98nBF8HotM7tuASSUudtR4CxoMS1GRuY/2hZ+tqi7UndJJL8eK7GZPSKkDNbqcTeKySSf2+QPZYmeAEGnnF9kkk3GlQTYo6gK3FTB66kjlwuAbGfY7FOZS3K6kg2Zx2eIM2ULZrlJJagQ9Tx52WCkiw90TgQb3SSUEZtZHFdDtVpZqKavjEeR7fEelx6LN1lQC420AK4kuxLmQc5XFFeZ93aKGXxacFxJUxSSSJ2Q5wNFxJJFRx//2Q==",
        "title": "Wheat crop close-up",
        "tag": "crop"
   
    }

]

device = "cpu"
model  = WheatNet(4, 0.4).to(device)
ckpt   = torch.load("wheat_model_v2.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Model loaded successfully!")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wheat Disease Detector</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #f7f5f0; --card: #ffffff; --border: #e8e4dc;
    --text: #1a1a1a; --muted: #6b6b6b;
    --accent: #2d5a27; --accent-lt: #eaf3e8;
  }
  body { font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 0; }

  /* NAV */
  nav { background: var(--accent); padding: 16px 32px; display: flex; align-items: center; justify-content: space-between; }
  nav h1 { font-family: 'DM Serif Display', serif; color: #fff; font-size: 1.5rem; font-weight: 400; letter-spacing: -0.3px; }
  nav span { color: rgba(255,255,255,0.7); font-size: 0.85rem; }

  /* HERO PHOTO GALLERY */
  .gallery { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; height: 220px; overflow: hidden; }
  .gallery-item { position: relative; overflow: hidden; }
  .gallery-item img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.4s ease; display: block; }
  .gallery-item:hover img { transform: scale(1.05); }
  .gallery-tag { position: absolute; bottom: 8px; left: 8px; background: rgba(0,0,0,0.55); color: #fff; font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 500; }

  /* MAIN LAYOUT */
  .main { max-width: 960px; margin: 0 auto; padding: 40px 20px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
  @media (max-width: 680px) { .two-col { grid-template-columns: 1fr; } .gallery { grid-template-columns: repeat(2,1fr); height: 180px; } }

  /* CARD */
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 28px; }
  .card-title { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted); margin-bottom: 18px; }

  /* UPLOAD */
  .upload-zone { border: 2px dashed var(--border); border-radius: 12px; padding: 40px 20px; text-align: center; cursor: pointer; transition: border-color 0.2s, background 0.2s; position: relative; }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--accent); background: var(--accent-lt); }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  .upload-icon { width: 44px; height: 44px; background: var(--accent-lt); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 14px; font-size: 20px; }
  .upload-zone h3 { font-size: 0.95rem; font-weight: 500; margin-bottom: 5px; }
  .upload-zone p { font-size: 0.82rem; color: var(--muted); }
  #preview-wrap { display: none; margin-top: 16px; }
  #preview-wrap img { width: 100%; max-height: 220px; object-fit: contain; border-radius: 10px; border: 1px solid var(--border); }

  /* BUTTON */
  .btn { display: block; width: 100%; margin-top: 16px; padding: 13px; background: var(--accent); color: #fff; border: none; border-radius: 10px; font-family: 'DM Sans', sans-serif; font-size: 0.92rem; font-weight: 500; cursor: pointer; transition: opacity 0.2s, transform 0.1s; }
  .btn:hover { opacity: 0.9; } .btn:active { transform: scale(0.98); } .btn:disabled { opacity: 0.45; cursor: not-allowed; }

  /* RESULT SECTION */
  #result { display: none; margin-top: 22px; padding-top: 22px; border-top: 1px solid var(--border); }
  .result-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 6px; }
  .result-disease { font-family: 'DM Serif Display', serif; font-size: 1.75rem; font-weight: 400; margin-bottom: 5px; }
  .result-desc { font-size: 0.86rem; color: var(--muted); line-height: 1.6; margin-bottom: 16px; }
  .confidence-badge { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 500; margin-bottom: 14px; }
  .severity-badge { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 600; margin-left: 8px; letter-spacing: 0.5px; }

  /* PROBABILITY BARS */
  .bars { display: flex; flex-direction: column; gap: 9px; }
  .bar-row { display: flex; align-items: center; gap: 10px; }
  .bar-label { font-size: 0.8rem; width: 120px; flex-shrink: 0; }
  .bar-track { flex: 1; height: 7px; background: var(--border); border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
  .bar-pct { font-size: 0.8rem; font-weight: 500; width: 38px; text-align: right; color: var(--muted); }

  /* RECOMMENDATION PANEL */
  #rec-panel { display: none; margin-top: 28px; }
  .rec-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
  .rec-header-dot { width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }
  .rec-header h2 { font-family: 'DM Serif Display', serif; font-size: 1.3rem; font-weight: 400; }
  .rec-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }
  .rec-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
  .rec-cat { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
  .rec-cat-icon { font-size: 16px; }
  .rec-cat-name { font-size: 0.82rem; font-weight: 600; color: var(--text); }
  .rec-steps { list-style: none; display: flex; flex-direction: column; gap: 8px; }
  .rec-steps li { font-size: 0.82rem; color: var(--muted); line-height: 1.5; padding-left: 14px; position: relative; }
  .rec-steps li::before { content: '—'; position: absolute; left: 0; color: var(--border); }

  /* GALLERY SECTION */
  .gallery-section { margin-top: 48px; }
  .gallery-section-title { font-family: 'DM Serif Display', serif; font-size: 1.5rem; font-weight: 400; margin-bottom: 6px; }
  .gallery-section-sub { font-size: 0.88rem; color: var(--muted); margin-bottom: 20px; }
  .photo-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  @media (max-width: 600px) { .photo-grid { grid-template-columns: repeat(2, 1fr); } }
  .photo-card { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); background: var(--card); }
  .photo-card img { width: 100%; height: 160px; object-fit: cover; display: block; transition: transform 0.3s; }
  .photo-card:hover img { transform: scale(1.04); }
  .photo-info { padding: 10px 12px; }
  .photo-info p { font-size: 0.82rem; font-weight: 500; margin-bottom: 2px; }
  .photo-tag { font-size: 0.72rem; color: var(--muted); }
  .photo-tag span { background: var(--accent-lt); color: var(--accent); padding: 2px 7px; border-radius: 4px; font-weight: 500; }

  /* SPINNER */
  .spinner { display: inline-block; width: 15px; height: 15px; border: 2px solid rgba(255,255,255,0.4); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; margin-right: 7px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* FOOTER */
  footer { text-align: center; padding: 32px 20px; font-size: 0.82rem; color: var(--muted); border-top: 1px solid var(--border); margin-top: 48px; }
</style>
</head>
<body>

<nav>
  <h1>🌾 Wheat Disease Detector</h1>
  <span>AI-powered · 96.51% accuracy</span>
</nav>

<!-- HERO GALLERY -->
<div class="gallery" id="heroGallery"></div>

<div class="main">

  <div class="two-col">

    <!-- LEFT: Upload -->
    <div class="card">
      <div class="card-title">Diagnose your crop</div>
      <div class="upload-zone" id="dropzone">
        <input type="file" id="fileInput" accept="image/*">
        <div class="upload-icon">🌿</div>
        <h3>Drop a leaf image here</h3>
        <p>or click to browse — JPG, PNG supported</p>
      </div>
      <div id="preview-wrap"><img id="preview" src="" alt="Preview"></div>
      <button class="btn" id="predictBtn" disabled>Analyse Image</button>

      <div id="result">
        <div class="result-label">Diagnosis</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
          <div class="result-disease" id="res-name"></div>
          <span class="severity-badge" id="res-severity"></span>
        </div>
        <span class="confidence-badge" id="res-badge"></span>
        <div class="result-desc" id="res-desc"></div>
        <div class="result-label" style="margin-bottom:10px">All probabilities</div>
        <div class="bars" id="res-bars"></div>
      </div>
    </div>

    <!-- RIGHT: Info panel -->
    <div style="display:flex;flex-direction:column;gap:16px;">
      <div class="card">
        <div class="card-title">About this system</div>
        <p style="font-size:0.88rem;color:var(--muted);line-height:1.7;">This AI model is trained on thousands of wheat leaf images to detect four conditions: <strong>Black Rust</strong>, <strong>Brown Rust</strong>, <strong>Yellow Rust</strong>, and <strong>Healthy Wheat</strong>. Upload a clear image of the affected leaf for best results.</p>
      </div>
      
     
      <div class="card">
        <div class="card-title">Disease severity guide</div>
        <div style="display:flex;flex-direction:column;gap:8px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="width:10px;height:10px;border-radius:50%;background:#c0392b;flex-shrink:0;"></span>
            <span style="font-size:0.83rem;"><strong>Black Rust</strong> — <span style="color:var(--muted);">High severity, acts fast</span></span>
          </div>
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="width:10px;height:10px;border-radius:50%;background:#e67e22;flex-shrink:0;"></span>
            <span style="font-size:0.83rem;"><strong>Brown Rust</strong> — <span style="color:var(--muted);">Medium severity</span></span>
          </div>
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="width:10px;height:10px;border-radius:50%;background:#d4ac0d;flex-shrink:0;"></span>
            <span style="font-size:0.83rem;"><strong>Yellow Rust</strong> — <span style="color:var(--muted);">High severity, spreads rapidly</span></span>
          </div>
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="width:10px;height:10px;border-radius:50%;background:#27ae60;flex-shrink:0;"></span>
            <span style="font-size:0.83rem;"><strong>Healthy</strong> — <span style="color:var(--muted);">No disease, maintain care</span></span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- RECOMMENDATION PANEL -->
  <div id="rec-panel">
    <div class="rec-header">
      <div class="rec-header-dot" id="rec-dot"></div>
      <h2 id="rec-title">Treatment Recommendations</h2>
    </div>
    <div class="rec-grid" id="rec-grid"></div>
  </div>

  <!-- WHEAT PHOTO GALLERY -->
  <div class="gallery-section">
    <div class="gallery-section-title">Wheat field gallery</div>
    <div class="gallery-section-sub">Reference photos of wheat crops, healthy leaves, and common disease symptoms.</div>
    <div class="photo-grid" id="photoGrid"></div>
  </div>

</div>

<footer>
  Wheat Disease Detector · Deep Learning Based Automated Diagnosis System · CNN + Logistic Regression · 96.51% Accuracy
</footer>

<script>
const CLASS_INFO = {{ class_info | tojson }};
const WHEAT_PHOTOS = {{ wheat_photos | tojson }};

const CLASS_COLORS = {
  "Black Rust": "#c0392b",
  "Brown Rust": "#e67e22",
  "Healthy Wheat": "#27ae60",
  "Yellow Rust": "#d4ac0d"
};

const SEVERITY_STYLES = {
  "High":   { bg: "#fdecea", color: "#c0392b" },
  "Medium": { bg: "#fef3e2", color: "#e67e22" },
  "None":   { bg: "#eaf7f0", color: "#27ae60" }
};

// Hero gallery (first 3 non-disease photos)
const heroEl = document.getElementById('heroGallery');
WHEAT_PHOTOS.slice(0, 4).forEach(p => {
  heroEl.innerHTML += `<div class="gallery-item">
    <img src="${p.url}" alt="${p.title}" onerror="this.parentElement.style.background='#e8e4dc'">
    <div class="gallery-tag">${p.tag}</div>
  </div>`;
});

// Photo grid
const photoGrid = document.getElementById('photoGrid');
WHEAT_PHOTOS.forEach(p => {
  photoGrid.innerHTML += `<div class="photo-card">
    <img src="${p.url}" alt="${p.title}" onerror="this.style.height='80px';this.style.background='#f0ede6'">
    <div class="photo-info">
      <p>${p.title}</p>
      <div class="photo-tag"><span>${p.tag}</span></div>
    </div>
  </div>`;
});

// Upload & predict
const fileInput   = document.getElementById('fileInput');
const preview     = document.getElementById('preview');
const previewWrap = document.getElementById('preview-wrap');
const predictBtn  = document.getElementById('predictBtn');
const dropzone    = document.getElementById('dropzone');
const resultEl    = document.getElementById('result');
const recPanel    = document.getElementById('rec-panel');

let selectedFile = null;

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  previewWrap.style.display = 'block';
  predictBtn.disabled = false;
  resultEl.style.display = 'none';
  recPanel.style.display = 'none';
}

fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => { e.preventDefault(); dropzone.classList.remove('drag'); handleFile(e.dataTransfer.files[0]); });

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  predictBtn.disabled = true;
  predictBtn.innerHTML = '<span class="spinner"></span>Analysing...';

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }

    const color = CLASS_COLORS[data.disease] || '#2d5a27';
    const info  = CLASS_INFO[data.disease];
    const sev   = SEVERITY_STYLES[info.severity] || SEVERITY_STYLES["None"];

    document.getElementById('res-name').textContent = data.disease;
    document.getElementById('res-name').style.color = color;
    document.getElementById('res-desc').textContent = data.description;

    const sevBadge = document.getElementById('res-severity');
    sevBadge.textContent = info.severity + ' severity';
    sevBadge.style.background = sev.bg;
    sevBadge.style.color = sev.color;

    const badge = document.getElementById('res-badge');
    badge.textContent = data.confidence + '% confidence';
    badge.style.background = color + '18';
    badge.style.color = color;

    const barsEl = document.getElementById('res-bars');
    barsEl.innerHTML = '';
    data.probabilities.forEach(item => {
      const c = CLASS_COLORS[item.class] || '#2d5a27';
      barsEl.innerHTML += `<div class="bar-row">
        <span class="bar-label">${item.class}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${item.prob}%;background:${c}"></div></div>
        <span class="bar-pct">${item.prob}%</span>
      </div>`;
    });

    resultEl.style.display = 'block';

    // Recommendations
    document.getElementById('rec-dot').style.background = color;
    document.getElementById('rec-title').textContent = data.disease + ' — Treatment & Recommendations';
    const recGrid = document.getElementById('rec-grid');
    recGrid.innerHTML = '';
    info.recommendations.forEach(rec => {
      const steps = rec.steps.map(s => `<li>${s}</li>`).join('');
      recGrid.innerHTML += `<div class="rec-card">
        <div class="rec-cat">
          <span class="rec-cat-icon">${rec.icon}</span>
          <span class="rec-cat-name">${rec.category}</span>
        </div>
        <ul class="rec-steps">${steps}</ul>
      </div>`;
    });
    recPanel.style.display = 'block';
    recPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch(err) {
    alert('Something went wrong. Is the Flask server running?');
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = 'Analyse Again';
  }
});
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    from flask import render_template_string
    return render_template_string(HTML, class_info=CLASS_INFO, wheat_photos=WHEAT_PHOTOS)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"})
    try:
        img    = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()
        pred_idx   = probs.argmax()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 1)
        probabilities = [
            {"class": CLASS_NAMES[i], "prob": round(float(p) * 100, 1)}
            for i, p in enumerate(probs)
        ]
        probabilities.sort(key=lambda x: -x["prob"])
        return jsonify({
            "disease"      : pred_class,
            "confidence"   : confidence,
            "description"  : CLASS_INFO[pred_class]["desc"],
            "probabilities": probabilities,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("\n" + "="*45)
    print("  Wheat Disease Detector running!")
    print("  Open http://127.0.0.1:5000 in browser")
    print("="*45 + "\n")
    app.run(debug=False)