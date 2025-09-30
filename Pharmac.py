import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EssentialOilPharmacopoeiaAnalyzer:
    def __init__(self, oil_name):
        self.oil = oil_name
        self.colors = ['#8B4513', '#228B22', '#FFD700', '#8A2BE2', '#FF6B6B', 
                      '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', '#2A9D8F']
        
        self.start_year = 2000
        self.end_year = 2025
        
        # Configuration spécifique pour chaque huile essentielle
        self.config = self._get_oil_config()
        
    def _get_oil_config(self):
        """Retourne la configuration spécifique pour chaque huile essentielle"""
        configs = {
            "Lavande": {
                "production_base": 150,  # tonnes annuelles
                "price_base": 45,  # €/kg
                "type": "relaxante",
                "proprietes": ["calmante", "cicatrisante", "antiseptique", "analgésique"],
                "regions": ["France", "Bulgarie", "Chine"],
                "rendement": 0.015  # 1.5% de rendement
            },
            "Menthe Poivrée": {
                "production_base": 80,
                "price_base": 60,
                "type": "tonique",
                "proprietes": ["digestive", "rafraichissante", "antalgique", "decongestionnante"],
                "regions": ["USA", "France", "Inde"],
                "rendement": 0.012
            },
            "Arbre à Thé": {
                "production_base": 120,
                "price_base": 35,
                "type": "antiseptique",
                "proprietes": ["antibacterienne", "antifongique", "antivirale", "immunostimulante"],
                "regions": ["Australie", "Chine", "Afrique du Sud"],
                "rendement": 0.020
            },
            "Eucalyptus": {
                "production_base": 200,
                "price_base": 25,
                "type": "respiratoire",
                "proprietes": ["expectorante", "decongestionnante", "antiseptique", "febrifuge"],
                "regions": ["Australie", "Chine", "Portugal"],
                "rendement": 0.018
            },
            "Ravintsara": {
                "production_base": 40,
                "price_base": 55,
                "type": "immunitaire",
                "proprietes": ["antivirale", "immunostimulante", "expectorante", "neurotonique"],
                "regions": ["Madagascar", "Comores"],
                "rendement": 0.008
            },
            "Palmarosa": {
                "production_base": 25,
                "price_base": 70,
                "type": "cosmetique",
                "proprietes": ["regenerante", "hydratante", "antibacterienne", "equilibrante"],
                "regions": ["Inde", "Nepal", "Indonesie"],
                "rendement": 0.006
            },
            "Ylang-Ylang": {
                "production_base": 30,
                "price_base": 85,
                "type": "aphrodisiaque",
                "proprietes": ["aphrodisiaque", "sedative", "hypotensive", "regulatrice"],
                "regions": ["Madagascar", "Comores", "Mayotte"],
                "rendement": 0.005
            },
            "Girofle": {
                "production_base": 60,
                "price_base": 40,
                "type": "antiseptique",
                "proprietes": ["antiseptique", "antalgique", "antiparasitaire", "stimulante"],
                "regions": ["Madagascar", "Indonesie", "Sri Lanka"],
                "rendement": 0.015
            },
            "Citron": {
                "production_base": 180,
                "price_base": 20,
                "type": "detoxifiante",
                "proprietes": ["antibacterienne", "detoxifiante", "tonique", "digestive"],
                "regions": ["Italie", "Espagne", "USA", "Argentine"],
                "rendement": 0.003
            },
            "Romarin": {
                "production_base": 90,
                "price_base": 38,
                "type": "tonique",
                "proprietes": ["tonique", "hepatique", "neurotonique", "antioxydante"],
                "regions": ["France", "Espagne", "Maroc", "Tunisie"],
                "rendement": 0.010
            },
            # Configuration par défaut
            "default": {
                "production_base": 50,
                "price_base": 50,
                "type": "polyvalente",
                "proprietes": ["antibacterienne", "antioxydante"],
                "regions": ["Divers"],
                "rendement": 0.010
            }
        }
        
        return configs.get(self.oil, configs["default"])
    
    def generate_pharmacopoeia_data(self):
        """Génère des données pour l'huile essentielle"""
        print(f"🌿 Génération des données pharmacologiques pour {self.oil}...")
        
        # Créer une base de données annuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='Y')
        
        data = {'Annee': [date.year for date in dates]}
        
        # Données de production et marché
        data['Production_Mondiale'] = self._simulate_global_production(dates)
        data['Prix_Moyen'] = self._simulate_average_price(dates)
        data['Demande_Mondiale'] = self._simulate_global_demand(dates)
        data['Surface_Cultivee'] = self._simulate_cultivation_area(dates)
        
        # Données de qualité et composition
        data['Teneur_Principes_Actifs'] = self._simulate_active_compounds(dates)
        data['Pureté_Chimique'] = self._simulate_chemical_purity(dates)
        data['Qualite_Bio'] = self._simulate_organic_quality(dates)
        
        # Données thérapeutiques
        data['Efficacite_Therapeutique'] = self._simulate_therapeutic_efficacy(dates)
        data['Etudes_Scientifiques'] = self._simulate_scientific_studies(dates)
        data['Demande_Therapeutique'] = self._simulate_therapeutic_demand(dates)
        
        # Applications et utilisations
        data['Usage_Aromatherapie'] = self._simulate_aromatherapy_use(dates)
        data['Usage_Cosmetique'] = self._simulate_cosmetic_use(dates)
        data['Usage_Pharmaceutique'] = self._simulate_pharmaceutical_use(dates)
        data['Usage_Alimentaire'] = self._simulate_food_use(dates)
        
        # Indicateurs économiques
        data['Valeur_Marche'] = self._simulate_market_value(dates)
        data['Croissance_Marche'] = self._simulate_market_growth(dates)
        data['Exportations'] = self._simulate_exports(dates)
        
        # Facteurs environnementaux
        data['Impact_Environnemental'] = self._simulate_environmental_impact(dates)
        data['Durabilite_Production'] = self._simulate_production_sustainability(dates)
        data['Rareté_Ressource'] = self._simulate_resource_scarcity(dates)
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances spécifiques
        self._add_essential_oil_trends(df)
        
        return df
    
    def _simulate_global_production(self, dates):
        """Simule la production mondiale (tonnes)"""
        base_production = self.config["production_base"]
        
        production = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year <= 2005:
                growth = 1 + 0.08 * i  # Croissance forte initiale
            elif 2006 <= year <= 2015:
                growth = 1 + 0.12 * (i-5)  # Expansion du marché
            elif 2016 <= year <= 2020:
                growth = 1 + 0.15 * (i-15)  # Boom du naturel
            else:
                growth = 1 + 0.10 * (i-20)  # Croissance soutenue
            
            noise = np.random.normal(1, 0.10)
            production.append(base_production * growth * noise)
        
        return production
    
    def _simulate_average_price(self, dates):
        """Simule le prix moyen (€/kg)"""
        base_price = self.config["price_base"]
        
        prices = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year <= 2005:
                variation = 1 + 0.03 * i  # Hausse modérée
            elif 2006 <= year <= 2012:
                variation = 1 + 0.05 * (i-5)  # Hausse due à la demande
            elif 2013 <= year <= 2018:
                variation = 1 + 0.08 * (i-12)  # Forte hausse qualité bio
            else:
                variation = 1 + 0.06 * (i-18)  # Hausse soutenue
            
            noise = np.random.normal(1, 0.08)
            prices.append(base_price * variation * noise)
        
        return prices
    
    def _simulate_global_demand(self, dates):
        """Simule la demande mondiale (tonnes)"""
        base_demand = self.config["production_base"] * 0.9
        
        demand = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year <= 2010:
                growth = 1 + 0.10 * i  # Demande croissante
            elif 2011 <= year <= 2020:
                growth = 1 + 0.14 * (i-10)  # Forte croissance
            else:
                growth = 1 + 0.12 * (i-20)  # Croissance soutenue
            
            noise = np.random.normal(1, 0.12)
            demand.append(base_demand * growth * noise)
        
        return demand
    
    def _simulate_cultivation_area(self, dates):
        """Simule la surface cultivée (hectares)"""
        base_area = self.config["production_base"] / self.config["rendement"] * 10
        
        areas = []
        for i, date in enumerate(dates):
            growth = 1 + 0.09 * i
            noise = np.random.normal(1, 0.15)
            areas.append(base_area * growth * noise)
        
        return areas
    
    def _simulate_active_compounds(self, dates):
        """Simule la teneur en principes actifs (%)"""
        compounds = []
        for i, date in enumerate(dates):
            base_level = 85  # Niveau de base en %
            
            year = date.year
            if year >= 2010:
                improvement = 1 + 0.005 * (year - 2010)  # Amélioration lente
            else:
                improvement = 1
            
            noise = np.random.normal(1, 0.04)
            compounds.append(base_level * improvement * noise)
        
        return compounds
    
    def _simulate_chemical_purity(self, dates):
        """Simule la pureté chimique (%)"""
        purity = []
        for i, date in enumerate(dates):
            base_purity = 92  # Pureté de base en %
            
            year = date.year
            if year >= 2008:
                improvement = 1 + 0.008 * (year - 2008)  # Amélioration des techniques
            else:
                improvement = 1
            
            noise = np.random.normal(1, 0.03)
            purity.append(base_purity * improvement * noise)
        
        return purity
    
    def _simulate_organic_quality(self, dates):
        """Simule la qualité bio (%)"""
        quality = []
        for i, date in enumerate(dates):
            base_quality = 60  # Part bio de base en %
            
            year = date.year
            if year >= 2015:
                growth = 1 + 0.025 * (year - 2015)  # Forte croissance du bio
            else:
                growth = 1
            
            noise = np.random.normal(1, 0.06)
            quality.append(base_quality * growth * noise)
        
        return quality
    
    def _simulate_therapeutic_efficacy(self, dates):
        """Simule l'efficacité thérapeutique (0-100)"""
        efficacy = []
        for i, date in enumerate(dates):
            base_efficacy = 75  # Efficacité de base
            
            year = date.year
            if year >= 2010:
                improvement = 1 + 0.012 * (year - 2010)  # Amélioration des connaissances
            else:
                improvement = 1
            
            noise = np.random.normal(1, 0.05)
            efficacy.append(base_efficacy * improvement * noise)
        
        return efficacy
    
    def _simulate_scientific_studies(self, dates):
        """Simule le nombre d'études scientifiques publiées"""
        studies = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year <= 2005:
                count = 5 + i * 2
            elif 2006 <= year <= 2015:
                count = 15 + (i-5) * 5
            else:
                count = 65 + (i-15) * 8
            
            noise = np.random.normal(1, 0.20)
            studies.append(count * noise)
        
        return studies
    
    def _simulate_therapeutic_demand(self, dates):
        """Simule la demande thérapeutique (0-100)"""
        demand = []
        for i, date in enumerate(dates):
            base_demand = 60
            
            year = date.year
            if year >= 2012:
                growth = 1 + 0.018 * (year - 2012)  # Croissance forte
            else:
                growth = 1
            
            noise = np.random.normal(1, 0.07)
            demand.append(base_demand * growth * noise)
        
        return demand
    
    def _simulate_aromatherapy_use(self, dates):
        """Simule l'usage en aromathérapie (0-100)"""
        usage = []
        for i, date in enumerate(dates):
            base_usage = 70
            
            year = date.year
            if year >= 2008:
                growth = 1 + 0.015 * (year - 2008)
            else:
                growth = 1
            
            noise = np.random.normal(1, 0.06)
            usage.append(base_usage * growth * noise)
        
        return usage
    
    def _simulate_cosmetic_use(self, dates):
        """Simule l'usage cosmétique (0-100)"""
        usage = []
        for i, date in enumerate(dates):
            base_usage = 65
            
            year = date.year
            if year >= 2010:
                growth = 1 + 0.020 * (year - 2010)  # Forte croissance
            else:
                growth = 1
            
            noise = np.random.normal(1, 0.08)
            usage.append(base_usage * growth * noise)
        
        return usage
    
    def _simulate_pharmaceutical_use(self, dates):
        """Simule l'usage pharmaceutique (0-100)"""
        usage = []
        for i, date in enumerate(dates):
            base_usage = 40
            
            year = date.year
            if year >= 2015:
                growth = 1 + 0.025 * (year - 2015)  # Croissance rapide
            else:
                growth = 1
            
            noise = np.random.normal(1, 0.10)
            usage.append(base_usage * growth * noise)
        
        return usage
    
    def _simulate_food_use(self, dates):
        """Simule l'usage alimentaire (0-100)"""
        usage = []
        for i, date in enumerate(dates):
            base_usage = 30
            
            year = date.year
            if year >= 2018:
                growth = 1 + 0.030 * (year - 2018)  # Très forte croissance
            else:
                growth = 1
            
            noise = np.random.normal(1, 0.12)
            usage.append(base_usage * growth * noise)
        
        return usage
    
    def _simulate_market_value(self, dates):
        """Simule la valeur de marché (millions €)"""
        base_value = self.config["production_base"] * self.config["price_base"] / 1000
        
        values = []
        for i, date in enumerate(dates):
            growth = 1 + 0.11 * i
            noise = np.random.normal(1, 0.13)
            values.append(base_value * growth * noise)
        
        return values
    
    def _simulate_market_growth(self, dates):
        """Simule la croissance du marché (%)"""
        growth_rates = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year <= 2005:
                rate = 8.0
            elif 2006 <= year <= 2015:
                rate = 12.0
            elif 2016 <= year <= 2020:
                rate = 15.0
            else:
                rate = 11.0
            
            noise = np.random.normal(1, 0.15)
            growth_rates.append(rate * noise)
        
        return growth_rates
    
    def _simulate_exports(self, dates):
        """Simule les exportations (tonnes)"""
        base_exports = self.config["production_base"] * 0.7
        
        exports = []
        for i, date in enumerate(dates):
            growth = 1 + 0.10 * i
            noise = np.random.normal(1, 0.14)
            exports.append(base_exports * growth * noise)
        
        return exports
    
    def _simulate_environmental_impact(self, dates):
        """Simule l'impact environnemental (0-100, plus bas = mieux)"""
        impact = []
        for i, date in enumerate(dates):
            base_impact = 45
            
            year = date.year
            if year >= 2010:
                improvement = 1 - 0.010 * (year - 2010)  # Amélioration
            else:
                improvement = 1
            
            noise = np.random.normal(1, 0.08)
            impact.append(base_impact * improvement * noise)
        
        return impact
    
    def _simulate_production_sustainability(self, dates):
        """Simule la durabilité de production (0-100)"""
        sustainability = []
        for i, date in enumerate(dates):
            base_sustainability = 65
            
            year = date.year
            if year >= 2012:
                improvement = 1 + 0.015 * (year - 2012)
            else:
                improvement = 1
            
            noise = np.random.normal(1, 0.07)
            sustainability.append(base_sustainability * improvement * noise)
        
        return sustainability
    
    def _simulate_resource_scarcity(self, dates):
        """Simule la rareté de la ressource (0-100, plus bas = mieux)"""
        scarcity = []
        for i, date in enumerate(dates):
            base_scarcity = 30
            
            year = date.year
            if year >= 2015:
                increase = 1 + 0.008 * (year - 2015)  # Légère augmentation
            else:
                increase = 1
            
            noise = np.random.normal(1, 0.10)
            scarcity.append(base_scarcity * increase * noise)
        
        return scarcity
    
    def _add_essential_oil_trends(self, df):
        """Ajoute des tendances spécifiques aux huiles essentielles"""
        for i, row in df.iterrows():
            year = row['Annee']
            
            # Début de popularité (2000-2005)
            if 2000 <= year <= 2005:
                df.loc[i, 'Usage_Aromatherapie'] *= 1.2
                df.loc[i, 'Etudes_Scientifiques'] *= 1.3
            
            # Reconnaissance scientifique (2006-2010)
            if 2006 <= year <= 2010:
                df.loc[i, 'Etudes_Scientifiques'] *= 1.5
                df.loc[i, 'Efficacite_Therapeutique'] *= 1.1
            
            # Boom du naturel (2011-2015)
            if 2011 <= year <= 2015:
                df.loc[i, 'Qualite_Bio'] *= 1.4
                df.loc[i, 'Demande_Mondiale'] *= 1.3
            
            # Intégration pharmaceutique (2016-2020)
            if 2016 <= year <= 2020:
                df.loc[i, 'Usage_Pharmaceutique'] *= 1.6
                df.loc[i, 'Prix_Moyen'] *= 1.2
            
            # Durabilité et éthique (2021-2025)
            if year >= 2021:
                df.loc[i, 'Durabilite_Production'] *= 1.2
                df.loc[i, 'Impact_Environnemental'] *= 0.9
                df.loc[i, 'Qualite_Bio'] *= 1.15
    
    def create_pharmacopoeia_analysis(self, df):
        """Crée une analyse complète de la pharmacopée"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Production et marché
        ax1 = plt.subplot(4, 2, 1)
        self._plot_production_market(df, ax1)
        
        # 2. Qualité et composition
        ax2 = plt.subplot(4, 2, 2)
        self._plot_quality_composition(df, ax2)
        
        # 3. Applications thérapeutiques
        ax3 = plt.subplot(4, 2, 3)
        self._plot_therapeutic_applications(df, ax3)
        
        # 4. Utilisations par secteur
        ax4 = plt.subplot(4, 2, 4)
        self._plot_usage_by_sector(df, ax4)
        
        # 5. Économie du marché
        ax5 = plt.subplot(4, 2, 5)
        self._plot_market_economics(df, ax5)
        
        # 6. Recherche scientifique
        ax6 = plt.subplot(4, 2, 6)
        self._plot_scientific_research(df, ax6)
        
        # 7. Durabilité environnementale
        ax7 = plt.subplot(4, 2, 7)
        self._plot_environmental_sustainability(df, ax7)
        
        # 8. Évolution globale
        ax8 = plt.subplot(4, 2, 8)
        self._plot_global_evolution(df, ax8)
        
        plt.suptitle(f'Analyse Pharmacopée - Huile Essentielle de {self.oil} ({self.start_year}-{self.end_year})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.oil}_pharmacopoeia_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Générer les insights
        self._generate_pharmacopoeia_insights(df)
    
    def _plot_production_market(self, df, ax):
        """Plot de la production et du marché"""
        ax.plot(df['Annee'], df['Production_Mondiale'], label='Production (tonnes)', 
               linewidth=2, color='#8B4513', alpha=0.8)
        ax.plot(df['Annee'], df['Demande_Mondiale'], label='Demande (tonnes)', 
               linewidth=2, color='#228B22', alpha=0.8)
        
        ax.set_title('Production et Demande Mondiales', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Tonnes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Second axe pour le prix
        ax2 = ax.twinx()
        ax2.plot(df['Annee'], df['Prix_Moyen'], label='Prix (€/kg)', 
                linewidth=2, color='#FFD700', linestyle='--')
        ax2.set_ylabel('Prix (€/kg)')
        ax2.legend(loc='upper right')
    
    def _plot_quality_composition(self, df, ax):
        """Plot de la qualité et composition"""
        ax.plot(df['Annee'], df['Teneur_Principes_Actifs'], label='Principes Actifs (%)', 
               linewidth=2, color='#8B4513', alpha=0.8)
        ax.plot(df['Annee'], df['Pureté_Chimique'], label='Pureté Chimique (%)', 
               linewidth=2, color='#228B22', alpha=0.8)
        ax.plot(df['Annee'], df['Qualite_Bio'], label='Qualité Bio (%)', 
               linewidth=2, color='#FFD700', alpha=0.8)
        
        ax.set_title('Qualité et Composition Chimique', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Pourcentage (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_therapeutic_applications(self, df, ax):
        """Plot des applications thérapeutiques"""
        ax.plot(df['Annee'], df['Efficacite_Therapeutique'], label='Efficacité Thérapeutique', 
               linewidth=2, color='#8B4513', alpha=0.8)
        ax.plot(df['Annee'], df['Demande_Therapeutique'], label='Demande Thérapeutique', 
               linewidth=2, color='#228B22', alpha=0.8)
        
        ax.set_title('Applications Thérapeutiques', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (0-100)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Second axe pour les études
        ax2 = ax.twinx()
        ax2.bar(df['Annee'], df['Etudes_Scientifiques'], label='Études Scientifiques', 
               alpha=0.3, color='#FF6B6B')
        ax2.set_ylabel('Nombre d\'Études')
        ax2.legend(loc='upper right')
    
    def _plot_usage_by_sector(self, df, ax):
        """Plot des utilisations par secteur"""
        years = df['Annee']
        width = 0.8
        
        bottom = np.zeros(len(years))
        categories = ['Usage_Aromatherapie', 'Usage_Cosmetique', 'Usage_Pharmaceutique', 'Usage_Alimentaire']
        colors = ['#8B4513', '#228B22', '#FFD700', '#8A2BE2']
        labels = ['Aromathérapie', 'Cosmétique', 'Pharmaceutique', 'Alimentaire']
        
        for i, category in enumerate(categories):
            ax.bar(years, df[category], width, label=labels[i], bottom=bottom, color=colors[i])
            bottom += df[category]
        
        ax.set_title('Répartition par Secteur d\'Utilisation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score d\'Utilisation (0-100)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_market_economics(self, df, ax):
        """Plot de l'économie du marché"""
        ax.plot(df['Annee'], df['Valeur_Marche'], label='Valeur de Marché (M€)', 
               linewidth=2, color='#8B4513', alpha=0.8)
        
        ax.set_title('Économie du Marché', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur (M€)', color='#8B4513')
        ax.tick_params(axis='y', labelcolor='#8B4513')
        ax.grid(True, alpha=0.3)
        
        # Second axe pour la croissance
        ax2 = ax.twinx()
        ax2.plot(df['Annee'], df['Croissance_Marche'], label='Croissance du Marché (%)', 
                linewidth=2, color='#228B22', linestyle='--')
        ax2.set_ylabel('Croissance (%)', color='#228B22')
        ax2.tick_params(axis='y', labelcolor='#228B22')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_scientific_research(self, df, ax):
        """Plot de la recherche scientifique"""
        ax.bar(df['Annee'], df['Etudes_Scientifiques'], label='Études Scientifiques', 
              color='#8B4513', alpha=0.7)
        
        ax.set_title('Recherche Scientifique', fontsize=12, fontweight='bold')
        ax.set_ylabel('Nombre d\'Études')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Second axe pour l'efficacité
        ax2 = ax.twinx()
        ax2.plot(df['Annee'], df['Efficacite_Therapeutique'], label='Efficacité Thérapeutique', 
                linewidth=2, color='#FF6B6B')
        ax2.set_ylabel('Efficacité (0-100)')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_environmental_sustainability(self, df, ax):
        """Plot de la durabilité environnementale"""
        ax.plot(df['Annee'], df['Durabilite_Production'], label='Durabilité Production', 
               linewidth=2, color='#228B22', alpha=0.8)
        ax.plot(df['Annee'], df['Impact_Environnemental'], label='Impact Environnemental', 
               linewidth=2, color='#FF6B6B', alpha=0.8)
        
        ax.set_title('Durabilité Environnementale', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (0-100)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Note: Impact environnemental plus bas = mieux
    
    def _plot_global_evolution(self, df, ax):
        """Plot de l'évolution globale"""
        # Normaliser les données pour superposition
        normalized_production = df['Production_Mondiale'] / df['Production_Mondiale'].max()
        normalized_price = df['Prix_Moyen'] / df['Prix_Moyen'].max()
        normalized_efficacy = df['Efficacite_Therapeutique'] / 100
        normalized_research = df['Etudes_Scientifiques'] / df['Etudes_Scientifiques'].max()
        
        ax.plot(df['Annee'], normalized_production, label='Production', linewidth=2)
        ax.plot(df['Annee'], normalized_price, label='Prix', linewidth=2)
        ax.plot(df['Annee'], normalized_efficacy, label='Efficacité', linewidth=2)
        ax.plot(df['Annee'], normalized_research, label='Recherche', linewidth=2)
        
        ax.set_title('Évolution Globale Comparative (Normalisée)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeurs Normalisées (0-1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _generate_pharmacopoeia_insights(self, df):
        """Génère des insights analytiques"""
        print(f"🌿 INSIGHTS PHARMACOPÉE - Huile Essentielle de {self.oil}")
        print("=" * 60)
        
        # 1. Statistiques de base
        print("\n1. 📊 STATISTIQUES GÉNÉRALES:")
        avg_production = df['Production_Mondiale'].mean()
        avg_price = df['Prix_Moyen'].mean()
        avg_market_value = df['Valeur_Marche'].mean()
        avg_studies = df['Etudes_Scientifiques'].mean()
        
        print(f"Production moyenne annuelle: {avg_production:.0f} tonnes")
        print(f"Prix moyen: {avg_price:.1f} €/kg")
        print(f"Valeur moyenne du marché: {avg_market_value:.1f} M€")
        print(f"Études scientifiques moyennes: {avg_studies:.0f}")
        
        # 2. Croissance
        print("\n2. 📈 TAUX DE CROISSANCE:")
        production_growth = ((df['Production_Mondiale'].iloc[-1] / 
                            df['Production_Mondiale'].iloc[0]) - 1) * 100
        market_growth = ((df['Valeur_Marche'].iloc[-1] / 
                         df['Valeur_Marche'].iloc[0]) - 1) * 100
        
        print(f"Croissance de la production ({self.start_year}-{self.end_year}): {production_growth:.1f}%")
        print(f"Croissance de la valeur du marché: {market_growth:.1f}%")
        
        # 3. Qualité et composition
        print("\n3. 🔬 QUALITÉ ET COMPOSITION:")
        avg_compounds = df['Teneur_Principes_Actifs'].mean()
        avg_purity = df['Pureté_Chimique'].mean()
        avg_bio = df['Qualite_Bio'].mean()
        
        print(f"Teneur moyenne en principes actifs: {avg_compounds:.1f}%")
        print(f"Pureté chimique moyenne: {avg_purity:.1f}%")
        print(f"Part moyenne de qualité bio: {avg_bio:.1f}%")
        
        # 4. Applications thérapeutiques
        print("\n4. 💊 APPLICATIONS THÉRAPEUTIQUES:")
        avg_efficacy = df['Efficacite_Therapeutique'].mean()
        total_studies = df['Etudes_Scientifiques'].sum()
        last_efficacy = df['Efficacite_Therapeutique'].iloc[-1]
        
        print(f"Efficacité thérapeutique moyenne: {avg_efficacy:.1f}/100")
        print(f"Total d'études scientifiques: {total_studies:.0f}")
        print(f"Dernière efficacité mesurée: {last_efficacy:.1f}/100")
        
        # 5. Spécificités de l'huile
        print(f"\n5. 🌟 SPÉCIFICITÉS DE L'HUILE DE {self.oil.upper()}:")
        print(f"Type: {self.config['type']}")
        print(f"Propriétés: {', '.join(self.config['proprietes'])}")
        print(f"Régions de production: {', '.join(self.config['regions'])}")
        print(f"Rendement: {self.config['rendement']*100:.1f}%")
        
        # 6. Évolutions marquantes
        print("\n6. 📅 ÉVOLUTIONS MARQUANTES:")
        print("• 2000-2005: Début de popularité et reconnaissance")
        print("• 2006-2010: Reconnaissance scientifique croissante")
        print("• 2011-2015: Boom des produits naturels et bio")
        print("• 2016-2020: Intégration dans l'industrie pharmaceutique")
        print("• 2021-2025: Focus sur la durabilité et l'éthique")
        
        # 7. Recommandations stratégiques
        print("\n7. 💡 RECOMMANDATIONS STRATÉGIQUES:")
        if "relaxante" in self.config["proprietes"]:
            print("• Développer les applications bien-être et relaxation")
            print("• Collaborer avec les centres de spa et thalassothérapie")
        if "antiseptique" in self.config["proprietes"]:
            print("• Promouvoir les usages en désinfection naturelle")
            print("• Développer les formulations pour soins cutanés")
        if "digestive" in self.config["proprietes"]:
            print("• Explorer les applications en gastro-entérologie")
            print("• Développer les compléments alimentaires naturels")
        
        print("• Investir dans la recherche clinique et scientifique")
        print("• Développer l'agriculture biologique et durable")
        print("• Renforcer la traçabilité et la qualité des produits")
        print("• Explorer les synergies avec d'autres huiles essentielles")
        print("• Développer les applications en médecine intégrative")

def main():
    """Fonction principale pour la pharmacopée des huiles essentielles"""
    # Liste des huiles essentielles
    huiles_essentielles = [
        "Lavande", "Menthe Poivrée", "Arbre à Thé", "Eucalyptus", "Ravintsara",
        "Palmarosa", "Ylang-Ylang", "Girofle", "Citron", "Romarin",
        "Tea Tree", "Géranium", "Camomille", "Sauge", "Niaouli",
        "Basilic", "Cèdre", "Encens", "Myrrhe", "Vetiver"
    ]
    
    print("🌿 ANALYSE PHARMACOPÉE DES HUILES ESSENTIELLES (2000-2025)")
    print("=" * 60)
    print("📊 Métriques: Production, qualité, applications thérapeutiques")
    print("💰 Économie: Marché, prix, croissance")
    print("🔬 Science: Recherche, efficacité, études")
    print("=" * 60)
    
    # Demander à l'utilisateur de choisir une huile
    print("Liste des huiles essentielles disponibles:")
    for i, huile in enumerate(huiles_essentielles, 1):
        print(f"{i}. {huile}")
    
    try:
        choix = int(input("\nChoisissez le numéro de l'huile essentielle à analyser: "))
        if choix < 1 or choix > len(huiles_essentielles):
            raise ValueError
        huile_selectionnee = huiles_essentielles[choix-1]
    except (ValueError, IndexError):
        print("Choix invalide. Sélection de Lavande par défaut.")
        huile_selectionnee = "Lavande"
    
    # Initialiser l'analyseur
    analyzer = EssentialOilPharmacopoeiaAnalyzer(huile_selectionnee)
    
    # Générer les données
    pharmacopoeia_data = analyzer.generate_pharmacopoeia_data()
    
    # Sauvegarder les données
    output_file = f'{huile_selectionnee}_pharmacopoeia_data_2000_2025.csv'
    pharmacopoeia_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(pharmacopoeia_data[['Annee', 'Production_Mondiale', 'Prix_Moyen', 'Efficacite_Therapeutique', 'Etudes_Scientifiques']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse pharmacopée...")
    analyzer.create_pharmacopoeia_analysis(pharmacopoeia_data)
    
    print(f"\n✅ Analyse pharmacopée de l'huile de {huile_selectionnee} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("📦 Données: Production, qualité, applications, recherche, durabilité")

if __name__ == "__main__":
    main()