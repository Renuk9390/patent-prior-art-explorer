"""
Curated demo corpus for the Patent Prior-Art Explorer.

Each entry is a REAL, published patent document (publicly disclosed by a
patent office, hence free of copyright restriction on the specification
text itself). Text below is drawn directly from each patent's published
abstract / description, lightly trimmed for length. Patent numbers and
URLs let a visitor jump to the full, authoritative document on Google
Patents at any time.

This file intentionally contains no logic -- it's pure data, so updating
the demo corpus later never requires touching the pipeline code.
"""

CORPUS = [
    {
        "id": "US20100201310A1",
        "title": "Wireless power transfer system",
        "domain": "Power Electronics",
        "url": "https://patents.google.com/patent/US20100201310A1/en",
        "paragraphs": [
            "Inductive coupling refers to the transfer of energy from one circuit "
            "component to another through a shared electromagnetic field. In "
            "inductive coupling, a current running in an emitting coil induces "
            "another current in a receiving coil. The two coils are in close "
            "proximity, but do not touch.",

            "Inductive coupling has been used in a variety of systems, including "
            "but not limited to systems that wirelessly charge a battery in a "
            "portable electronic device. In such systems, the portable electronic "
            "device is placed in close proximity to a charging station. A first "
            "induction coil in the charging station is used to create an "
            "alternating electromagnetic field, and a second induction coil in "
            "the portable electronic device derives power from the electromagnetic "
            "field and converts it back into electrical current to charge the "
            "battery, so there is no need for direct electrical contact between "
            "the battery and the charging station.",

            "A method for wirelessly transferring power from a charging station "
            "to a portable electronic device is described herein. A wireless "
            "communication link is established with the portable electronic "
            "device, and payment information is then received from the device "
            "via that link. Responsive to receiving the payment information, "
            "power is transferred to the portable electronic device over a "
            "wireless power link, which may be established in accordance with "
            "a Near Field Communication protocol, a Bluetooth protocol, a "
            "ZigBee protocol, or an IEEE 802.11 protocol.",
        ],
    },
    {
        "id": "US9490507B2",
        "title": "Li-ion battery thermal runaway suppression system using "
        "microchannel coolers and refrigerant injections",
        "domain": "Energy Storage & Safety",
        "url": "https://patents.google.com/patent/US9490507B2/en",
        "paragraphs": [
            "A battery management system with thermally integrated fire "
            "suppression includes a multiplicity of individual battery cells "
            "in a housing, a multiplicity of cooling passages in the housing "
            "within or between the cells, a multiplicity of sensors operably "
            "connected to the individual battery cells and adapted to detect a "
            "thermal runaway event, and a management system adapted to inject "
            "coolant into the cooling passages upon detection of the thermal "
            "runaway event, so that the event is rapidly quenched.",

            "The present invention provides a method of thermally managing a "
            "battery pack from thermal runaway events. The method includes "
            "providing a plurality of cooling passages between the individual "
            "battery cells, providing a source of coolant operatively connected "
            "to one or more of the cooling passages, providing a sensor "
            "operably connected to each of the battery cells and adapted to "
            "detect a thermal runaway event, and injecting coolant from the "
            "source into at least one cooling passage upon detection of the "
            "event by the sensor.",

            "In one embodiment, the system includes a remote lithium-ion "
            "battery management system that is operatively connected to the "
            "sensors and the coolant injection mechanism, allowing the thermal "
            "runaway response to be coordinated across the entire battery pack "
            "rather than at a single cell in isolation.",
        ],
    },
    {
        "id": "US9445184B2",
        "title": "Active noise reduction headphone",
        "domain": "Audio Signal Processing",
        "url": "https://patents.google.com/patent/US9445184/en",
        "paragraphs": [
            "An active noise reduction earphone includes a speaker, a plurality "
            "of microphones, and a feedback system. Each microphone is "
            "displaced from the speaker and from the other microphones, and "
            "each microphone generates a microphone signal responsive to "
            "received acoustic noise. The feedback system receives a "
            "combination of the microphone signals and generates an inverse "
            "noise signal that is applied to the speaker, which in turn "
            "generates an inverse acoustic noise signal that substantially "
            "cancels the acoustic noise at a predetermined location relative "
            "to the speaker and the microphones.",

            "The feedback system can include a microphone signal combiner in "
            "communication with the microphones. The microphone signal "
            "combiner generates a signal that may be a sum or weighted sum of "
            "the microphone signals, used to generate the inverse noise "
            "signal. The earphone has an increased noise reduction bandwidth "
            "and improved cancellation capability relative to conventional "
            "earphones.",

            "Feedback-based active noise reduction headphones typically "
            "utilize a feedback system that includes a microphone positioned "
            "near the ear of a user and also near the earphone speaker. A "
            "feedback circuit attempts to reduce the energy in the microphone "
            "signal generated by acoustic noise to zero, generating a "
            "compensating signal that is 180 degrees out of phase with the "
            "sensed noise signal.",
        ],
    },
    {
        "id": "US20250117955A1",
        "title": "Obstacle avoidance using a monocular camera in a vehicle",
        "domain": "Autonomous Systems & Computer Vision",
        "url": "https://patents.google.com/patent/US20250117955A1/en",
        "paragraphs": [
            "Obstacle avoidance is a critical task for unmanned aerial "
            "vehicles (UAVs) and semi-autonomous vehicles to operate safely "
            "in complex environments. Traditional obstacle avoidance systems "
            "rely on multiple sensors, such as LiDAR and depth cameras, which "
            "can be expensive, power-consuming, and bulky.",

            "Monocular cameras are a more affordable, lightweight, and "
            "compact alternative for obstacle avoidance. However, monocular "
            "cameras only provide two-dimensional image information, making "
            "it challenging to accurately estimate the distance to obstacles "
            "and make precise obstacle avoidance decisions.",

            "Recent advances in deep learning have made it possible to "
            "estimate depth from monocular images, allowing a vehicle to "
            "infer three-dimensional obstacle positions from a single, "
            "inexpensive camera rather than relying on a dedicated "
            "depth-sensing sensor such as LiDAR.",
        ],
    },
    {
        "id": "US20180090720A1",
        "title": "Flexible OLED Display Module",
        "domain": "Display Technology",
        "url": "https://patents.google.com/patent/US20180090720A1/en",
        "paragraphs": [
            "A flexible OLED display module that is capable of repetitive "
            "flexing and having a low radius of curvature is provided. "
            "According to an embodiment, a flexible OLED display module may "
            "include a first stack having a substrate, a backplane disposed "
            "on the substrate, and an organic electroluminescent layer formed "
            "on the backplane.",

            "The flexible OLED display module may further include a second "
            "stack having a lid layer and a polarizer deposited on the lid "
            "layer, with the first stack laminated to the second stack so "
            "that the device can be repeatedly folded without delamination "
            "or damage to the electroluminescent layer.",

            "Opto-electronic devices that make use of organic materials are "
            "becoming increasingly desirable for a number of reasons, "
            "including their suitability for thin, lightweight, and "
            "mechanically flexible display constructions not achievable with "
            "rigid glass-substrate displays.",
        ],
    },
    {
        "id": "US7658728B2",
        "title": "Microneedle array, patch, and applicator for transdermal "
        "drug delivery",
        "domain": "Medical Devices & Drug Delivery",
        "url": "https://patents.google.com/patent/US7658728B2/en",
        "paragraphs": [
            "Microneedle arrays and drug delivery devices are provided for "
            "transdermal administration of a drug formulation. An applicator "
            "device includes a rigid housing having a substantially planar "
            "application side, a recess in the housing in which a drug "
            "delivery patch that includes a microneedle array is disposed, "
            "and a button that can be depressed to drive the microneedles "
            "into the skin, piercing the stratum corneum.",

            "The housing further includes a roller used for completely "
            "inserting the microneedles into the biological barrier and for "
            "establishing a secure bond between the drug delivery device and "
            "the surface of the skin.",

            "Because the length of a hollow microneedle's central bore is "
            "much greater than its diameter, diffusional transport of drug "
            "through the bore may be unacceptably slow -- potentially even "
            "slower than diffusion through the stratum corneum in the "
            "absence of the microneedle altogether -- motivating the need "
            "for an improved microneedle array design.",
        ],
    },
]
