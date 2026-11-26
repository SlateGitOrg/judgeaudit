"""Question templates, answer sentences and neutral filler for the synthetic retailer data.

Answer text is only ever read by a real (HTTP) judge; the simulated judge reads the latent quality
and the word count. The text is still built to look like support answers, and the filler sentences
are deliberately information-free so that padding an answer cannot change its quality.
"""

from __future__ import annotations

import numpy as np

CATEGORIES = [
    "electronics specs",
    "apparel sizing",
    "returns",
    "shipping",
    "warranty",
    "compatibility",
    "care instructions",
    "availability",
]

PRODUCTS = {
    "electronics specs": ["the Orbit X2 laptop", "the Lumen 4K monitor", "the Pulse earbuds"],
    "apparel sizing": ["the Alder wool coat", "the Fenn running shorts", "the Harbor denim jacket"],
    "returns": ["the Cask espresso machine", "the Nimbus desk lamp", "the Trail hiking boots"],
    "shipping": [
        "the Vista standing desk",
        "the Ridge camping stove",
        "the Quill mechanical keyboard",
    ],
    "warranty": ["the Aero air fryer", "the Strata robot vacuum", "the Halo smart speaker"],
    "compatibility": ["the Volt 65W charger", "the Mesa docking station", "the Echo microphone"],
    "care instructions": [
        "the Linden linen shirt",
        "the Cove cast-iron pan",
        "the Moss leather bag",
    ],
    "availability": ["the Tide water bottle", "the Spire tripod", "the Gale desk fan"],
}

QUESTIONS = {
    "electronics specs": [
        "How much memory does {p} ship with?",
        "What is the battery life of {p} under normal use?",
        "Does {p} support fast charging, and at what wattage?",
        "What ports does {p} have?",
    ],
    "apparel sizing": [
        "Does {p} run small or large compared with usual sizing?",
        "Which size should I order for {p} if I am between two sizes?",
        "Is there a size chart for {p} with measurements in centimetres?",
        "Can you tell me the model's height and the size they wear for {p}?",
    ],
    "returns": [
        "How long do I have to return {p}?",
        "Can I return {p} if I have already opened the box?",
        "Who pays for return shipping on {p}?",
        "How soon is my refund issued after I send back {p}?",
    ],
    "shipping": [
        "When will {p} arrive if I order today?",
        "Is express delivery available for {p}?",
        "Do you ship {p} to a post office box?",
        "Can I change the delivery address for {p} after ordering?",
    ],
    "warranty": [
        "How long is the warranty on {p}?",
        "Does the warranty on {p} cover accidental damage?",
        "How do I make a warranty claim for {p}?",
        "Is the warranty on {p} transferable if I give it away?",
    ],
    "compatibility": [
        "Is {p} compatible with a ten year old computer?",
        "Will {p} work with both iOS and Android devices?",
        "Can I use {p} with a third party cable?",
        "Does {p} work with the previous generation accessories?",
    ],
    "care instructions": [
        "How should I wash {p}?",
        "Can {p} go in the dryer?",
        "What is the best way to remove a stain from {p}?",
        "How do I store {p} when it is not in use?",
    ],
    "availability": [
        "When will {p} be back in stock?",
        "Is {p} available in other colours?",
        "Can I reserve {p} at a nearby store?",
        "Will {p} be restocked before the holidays?",
    ],
}

ANSWER_SENTENCES = {
    "electronics specs": [
        "The specification sheet on the product page lists the exact figures for {p}.",
        "Figures are measured by the manufacturer under standard test conditions.",
        "Real world results depend on screen brightness and the workloads you run.",
        "The base configuration of {p} is the one shown in the first product photo.",
        "Higher configurations are listed as separate options at checkout.",
        "Firmware updates can change some of these figures after purchase.",
        "The box includes a quick start guide with the full port and power details.",
        "If you need a number for a purchase order we can send a signed data sheet.",
        "We recommend comparing the sheet with your own requirements before ordering.",
        "Our support team can confirm any single line on request.",
    ],
    "apparel sizing": [
        "{p} follows the size chart shown beside the size selector.",
        "Most customers find that it fits true to the chart.",
        "If you are between two sizes we suggest the larger one for a relaxed fit.",
        "Measurements on the chart are taken flat and given in centimetres and inches.",
        "The fabric has a little stretch, so it will ease slightly with wear.",
        "Layering over a thick sweater is easier one size up.",
        "Sleeve and inseam lengths are listed separately for each size.",
        "The model in the photos is 180 cm tall and wears a medium.",
        "Reviews from other customers often mention fit and are worth a look.",
        "Exchanges for a different size are free within the return window.",
    ],
    "returns": [
        "You can return {p} within 30 days of the delivery date.",
        "The item should be in its original packaging with all accessories included.",
        "Opened items are accepted if they are complete and show no signs of damage.",
        "Start the return from your order history and print the prepaid label.",
        "Return shipping is free when the item arrived faulty or was sent in error.",
        "For change of mind returns a flat fee is deducted from the refund.",
        "Refunds go back to the original payment method.",
        "Once the warehouse scans the parcel the refund is usually issued in three to five days.",
        "Gift returns are credited as store credit to the recipient.",
        "Final sale items are marked on the product page and cannot be returned.",
    ],
    "shipping": [
        "Standard delivery of {p} takes three to five working days.",
        "Orders placed before 2 pm are dispatched the same day.",
        "Express delivery is offered at checkout where the carrier covers your postcode.",
        "A tracking link is emailed as soon as the parcel leaves the warehouse.",
        "Bulky items are delivered by a two person team who will call ahead.",
        "We cannot deliver to post office boxes for items above a certain weight.",
        "The delivery address can be changed until the order is packed.",
        "Delays caused by severe weather are shown on the tracking page.",
        "Signature on delivery is required for orders over a set value.",
        "Customs duties for international orders are shown before you pay.",
    ],
    "warranty": [
        "{p} comes with a two year manufacturer warranty from the date of purchase.",
        "The warranty covers defects in materials and workmanship.",
        "Accidental damage and normal wear are not covered unless you add a protection plan.",
        "To make a claim, send your order number and a short description of the fault.",
        "We may ask for a photo or a short video showing the problem.",
        "Approved claims are repaired, replaced or refunded at our discretion.",
        "Keep your proof of purchase, because it is needed for any claim.",
        "Repairs normally take seven to ten working days including transit.",
        "The remaining warranty passes to a new owner when you show the original receipt.",
        "Statutory consumer rights apply in addition to this warranty.",
    ],
    "compatibility": [
        "{p} follows the common industry standard for this kind of accessory.",
        "It works with most devices made in the last several years.",
        "Very old devices may need a software update before they are recognised.",
        "Third party cables work if they are certified for the same standard.",
        "The compatibility list on the product page names the tested models.",
        "Some features need the companion app, which is free to download.",
        "Accessories from the previous generation connect in the same way.",
        "Performance can be limited by the slowest part of the chain.",
        "If your device is not on the list our team can check it for you.",
        "We will accept a return if the item turns out not to work with your setup.",
    ],
    "care instructions": [
        "Please follow the care label sewn into {p}.",
        "Wash it on a gentle cycle at 30 degrees with similar colours.",
        "Avoid fabric softener because it can coat the fibres over time.",
        "Reshape while damp and dry flat away from direct heat.",
        "Tumble drying is not recommended and can shrink the material.",
        "Treat stains early with cold water and a small amount of mild soap.",
        "Do not use bleach or harsh solvents on any part of the product.",
        "Store it somewhere cool and dry, folded loosely or on a wide hanger.",
        "A soft brush removes surface dust between washes.",
        "Following these steps will help it keep its shape and colour for years.",
    ],
    "availability": [
        "{p} is currently out of stock in the online store.",
        "The next delivery from the supplier is expected in two to three weeks.",
        "Dates can move if the supplier changes its schedule.",
        "You can sign up for a back in stock email on the product page.",
        "Other colours are shown under the colour selector when they are available.",
        "Store stock changes through the day, so the page shows a live count per store.",
        "A reserved item is held for 48 hours at the store of your choice.",
        "Popular items sell out quickly during seasonal sales.",
        "We cannot guarantee restocks before a given date.",
        "Pre-orders open on the product page once the supplier confirms a ship date.",
    ],
}

# Detail sentences appended to long answers; they apply to any product.
EXTRA_DETAIL = [
    "You can find the full terms on the help centre page for this topic.",
    "Details can differ slightly between regions, so check the page for your country.",
    "If anything here does not match what you see at checkout, the checkout page is correct.",
    "Business customers can ask for a written confirmation to attach to a purchase order.",
    "Our team is available by chat from eight in the morning to eight in the evening.",
    "Order numbers speed things up, so keep yours handy when you get in touch.",
    "We update this guidance whenever the supplier or the carrier changes its policy.",
    "Customer reviews on the product page often cover this question from experience.",
]

# Neutral filler: information-free sentences used to pad an answer without changing its content.
FILLER_BANK = [
    "Thank you for getting in touch with us today.",
    "We appreciate your patience and your interest in our products.",
    "Please do not hesitate to reach out if you have any other questions.",
    "We hope this information is helpful to you.",
    "Our team is always happy to assist with anything else you might need.",
    "We value your feedback and use it to improve our service.",
    "Thanks again for shopping with us and for choosing our store.",
    "It is a pleasure to help you with your enquiry.",
    "We look forward to serving you again in the near future.",
    "Customer satisfaction is important to everyone on our team.",
    "Feel free to contact us at any time if anything is unclear.",
    "We are glad you asked and we are here to help.",
    "Have a wonderful day and happy shopping.",
    "Your support means a great deal to our small team.",
    "Let us know if there is anything else we can do for you.",
    "We always welcome the chance to hear from our customers.",
]


def build_text_family(
    category: str, product: str, lengths: list[int], rng: np.random.Generator
) -> list[str]:
    """Answers of several lengths that share one sentence order (longer = shorter + more)."""
    sentences = [s.format(p=product) for s in ANSWER_SENTENCES[category]]
    order = rng.permutation(len(sentences))
    pool = [sentences[i] for i in order]
    pool += [EXTRA_DETAIL[i] for i in rng.permutation(len(EXTRA_DETAIL))]
    return [_take_words(pool, n) for n in lengths]


def _take_words(pool: list[str], n_words: int) -> str:
    words: list[str] = []
    i = 0
    while len(words) < n_words:
        words.extend(pool[i % len(pool)].split())
        i += 1
    words = words[: max(n_words, 3)]
    words[-1] = words[-1].rstrip(".,;:!?") + "."
    return " ".join(words)


def pad_to(text: str, target_words: int, rng: np.random.Generator) -> str:
    """Append neutral filler sentences to ``text`` until it has exactly ``target_words`` words.

    The last filler sentence is cut to fit, so the padded length is exact. If the text is already
    long enough it is returned unchanged.
    """
    words = text.split()
    if len(words) >= target_words:
        return text
    order = rng.permutation(len(FILLER_BANK))
    i = 0
    while len(words) < target_words:
        words.extend(FILLER_BANK[order[i % len(order)]].split())
        i += 1
    words = words[:target_words]
    words[-1] = words[-1].rstrip(".,;:!?") + "."
    return " ".join(words)
