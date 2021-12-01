from numpy.core.defchararray import count
import pandas as pd


def get_country(affiliation):
    country = None
    if type(affiliation) != str:
        return None
    # USA #101 North America
    if affiliation in ["IBM Research", "AT&T Inc", "Yahoo Research", "Google", "Brown University", "Carnegie Mellon University", "Facebook", "Google Research"] or ('USA' in affiliation) or ('State' in affiliation) or ('Chicago' in affiliation) or ('Georgia' in affiliation) or ('Pennsylvania' in affiliation) or ('Massachusetts' in affiliation) or ('Stanford' in affiliation) or ('Harvard' in affiliation) or ('New York' in affiliation) or ('California' in affiliation) or ('Ohio' in affiliation) or ('Maryland' in affiliation) or ('Utah' in affiliation) or ('Seattle' in affiliation) or ('Virginia' in affiliation) or ('Arizona' in affiliation) or ('Michigan' in affiliation) or ('Duke' in affiliation) or ('Palo Alto' in affiliation):
        country = 'USA'
    # 102 UK/Ireland
    if ('UK' in affiliation) or ('London' in affiliation) or ('Bedfordshire' in affiliation) or ('Scotland' in affiliation) or ('Oxford' in affiliation) or ('Aberdeen' in affiliation) or ('Manchester' in affiliation):
        country = 'UK'
    # Germany #103 Europe
    if ('Germany' in affiliation) or ('Berlin' in affiliation) or ('Munich' in affiliation) or ('München' in affiliation) or ('Saarland' in affiliation) or ('Hagen' in affiliation) or ('Darmstadt' in affiliation) or ('Reutlingen' in affiliation) or ('Kaiserslautern' in affiliation) or ('Karlsruhe' in affiliation) or ('Freiburg' in affiliation) or ('Stuttgart' in affiliation) or ('Aachen' in affiliation) or ('Dresden' in affiliation) or ('Tübingen' in affiliation):
        country = 'Germany'
    # France
    if ('France' in affiliation) or ('Avignon' in affiliation) or ('Paris' in affiliation) or ('Versailles' in affiliation) or ('Toulouse' in affiliation) or ('Marseille' in affiliation) or ('Nancy' in affiliation):
        country = 'France'
    # Italy
    if ('Italy' in affiliation) or ('Rome' in affiliation) or ('Turin' in affiliation) or ('Milan' in affiliation) or ('Padua' in affiliation) or ('Pisa' in affiliation):
        country = 'Italy'
    # China #122 Asia
    if ('China' in affiliation) or ('Hong Kong' in affiliation) or ('Chinese' in affiliation) or ('Peking' in affiliation) or ('Zhejiang' in affiliation):
        country = 'China'
    # Australia
    if ('Australia' in affiliation) or ('Queensland' in affiliation):
        country = 'Australia'
    # Canada
    if ('Canada' in affiliation) or ('Toronto' in affiliation) or ('Waterloo' in affiliation):
        country = 'Canada'
    # Netherlands
    if 'Netherlands' in affiliation:
        country = 'Netherlands'
    # South America
    if ('Brazil' in affiliation):
        country = 'Brazil'
    if ('Chile' in affiliation):
        country = 'Chile'
    if ('Venezuela' in affiliation):
        country = 'Venezuela'
    if ('Mexico' in affiliation):
        country = 'Mexico'
    # Switzerland
    if ('Switzerland' in affiliation) or ('Zurich' in affiliation) or ('Zürich' in affiliation):
        country = 'Switzerland'
    # Israel #123 Middle East
    if ('Israel' in affiliation) or ('Tel Aviv' in affiliation):
        country = 'Israel'
    # Singapore #122 Asia
    if ('Singapore' in affiliation) or ('Nanyang' in affiliation):
        country = 'Singapore'
    # India #122 Asia
    if ('India' in affiliation) or ('Bombay' in affiliation):
        country = 'India'
    # Spain
    if ('Spain' in affiliation) or ('Madrid' in affiliation) or ('Catalunya' in affiliation):
        country = 'Spain'
    # Austria
    if ('Austria' in affiliation) or ('Vienna' in affiliation):
        country = 'Austria'
    # UK/Ireland
    if 'Ireland' in affiliation:
        country = 'Ireland'
    # Greece
    if ('Greece' in affiliation) or ('Hellas' in affiliation) or ('Athens' in affiliation) or ('Athena' in affiliation) or ('Piraeus' in affiliation) or ('Crete' in affiliation):
        country = 'Greece'
    # Japan #122 Asia
    if ('Japan' in affiliation) or ('Tokyo' in affiliation) or ('Kyoto' in affiliation):
        country = 'Japan'
    # Scandinavia
    if ('Norway' in affiliation):
        country = 'Norway'
    if ('Sweden' in affiliation):
        country = 'Sweden'
    if ('Finland' in affiliation) or ('Helsinki' in affiliation):
        country = 'Finland'
    # otherEU: Denmark, Belgium, Portugal, Hungary, Czech, Poland, Slovenia, Luxembourg, Estonia, Bratislava
    if ('Denmark' in affiliation) or ('Luxembourg' in affiliation) or ('Estonia' in affiliation) or ('Bratislava' in affiliation):
        country = 'Denmark'
    if ('Belgium' in affiliation):
        country = 'Belgium'
    if ('Portugal' in affiliation):
        country = 'Portugal'
    if ('Sweden' in affiliation):
        country = 'Sweden'
    if ('Hungary' in affiliation):
        country = 'Hungary'
    if ('Czech' in affiliation):
        country = 'Czech'
    if ('Poland' in affiliation):
        country = 'Poland'
    if ('Slovenia' in affiliation):
        country = 'Slovenia'
    if ('Luxembourg' in affiliation):
        country = 'Luxembourg'
    if ('Estonia' in affiliation) or ('Bratislava' in affiliation):
        country = 'Estonia'
    if ('Albany' in affiliation):
        country = 'Albany'
    if ('New Zealand' in affiliation):
        country = 'New Zealand'
    if ('Russia' in affiliation):
        country = 'Russia'
    # otherAsia: Korea, Taiwan
    if ('Korea' in affiliation):
        country = 'Korea'
    if ('Taiwan' in affiliation):
        country = 'Taiwan'
    # Middle East: Qatar, Arabia, UAE, Turkey, Egypt, Iran
    if ('Qatar' in affiliation):
        country = 'Qatar'
    if ('Arabia' in affiliation):
        country = 'Arabia'
    if ('UAE' in affiliation):
        country = 'UAE'
    if ('Turkey' in affiliation):
        country = 'Turkey'
    if ('Egypt' in affiliation):
        country = 'Egypt'
    if ('Iran' in affiliation):
        country = 'Iran'

    return country


dem = pd.read_csv("demographics.csv")
dem["country"] = dem.affiliation.apply(get_country)
