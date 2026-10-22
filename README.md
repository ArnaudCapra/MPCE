# Structure à terme du skew : Rough Heston et MPCE

## 1. Objectif et motivation

Ce projet cherche à déterminer si rendre l’exposant de Hurst variable permet de reproduire, avec un seul facteur de volatilité, la structure à terme du skew observée sur le marché, notamment le changement de vitesse de sa décroissance selon la maturité.

Le point de départ est la régularité de la volatilité. Les travaux de Bennedsen, Lunde et Pakkanen, *Decoupling the short- and long-term behavior of stochastic volatility* (2016, publication en 2021), motivent une volatilité rugueuse dont la régularité peut varier dans le temps. Les études d’Alvarez-Ramirez et al. (2008) et de Cadoni, Melis et Trudda (2015) examinent également des exposants variables sur les séries de prix ; leur objet est distinct de la régularité de la volatilité étudiée ici.

Ce qui a motivé le passage des modèles à volatilité constante à ceux de volatilité stochastique est le constat que les rendements empiriques sont leptokurtiques et asymétriques. Plutôt que de remplacer le bruit brownien par une loi fidèle à la distribution empirique des rendements, on modélise la variation de l’amplitude du bruit. Le projet prolonge cette idée : après l’amplitude, c’est la régularité de la volatilité qui devient aléatoire.

L’amplitude et la régularité doivent être distinguées. La vol-of-vol gouverne l’ampleur des fluctuations ; l’exposant de Hurst caractérise leur régularité locale. Les construire séparément permet ensuite d’étudier leur interaction, notamment par la corrélation de leurs innovations. Le terme « multifractionnaire » désigne ici la variation temporelle de cet exposant, et non un spectre multifractal.

Le cadre de Loboda, Mies et Steland (2021), *Regularity of multifractional moving average processes with random Hurst exponent*, guide le choix d’un noyau adapté : l’objectif est que H représente la régularité de la volatilité, sans que celle-ci soit dictée par la rugosité du processus H lui-même. La normalisation vise à distinguer l’amplitude de la régularité ; cette distinction reste un objectif à vérifier dans le modèle.

Le prix conserve une dynamique brownienne, tandis que la volatilité porte la mémoire fractionnaire. À partir de Rough Heston, associé aux travaux d’El Euch et Rosenbaum, le MPCE remplace l’exposant constant par une diffusion de Jacobi corrélée. Ce choix, discuté notamment par Gouriéroux et Valéry (2004), offre un cadre borné et une loi stationnaire bêta sous les conditions appropriées. Les conditions aux frontières sont étudiées notamment par D’Onofrio, Tamborrino et Lansky (2018).

La comparaison porte donc sur deux constructions : une variance à exposant fixe et un facteur de volatilité à exposant variable. Le MPCE ajoute un état aléatoire H, mais ne superpose pas deux facteurs de volatilité. La question est de savoir si cette variation de régularité suffit à produire la rupture du skew de marché. Elle constitue l’objet du projet, pas une conclusion présupposée.

## 2. Les modèles en temps continu

Les deux modèles décrivent un sous-jacent \(S_t\) et sa variance instantanée \(V_t\). Le taux \(r\), le dividende \(q\) et le temps sont exprimés sur une base annuelle. Les browniens sont notés \(W\), et \(\Gamma\) désigne la fonction gamma.

### Rough Heston : une variance à mémoire d’exposant fixe

Le système s’écrit :

$$
\begin{aligned}
\frac{dS_t}{S_t}
&=(r-q)\,dt+\sqrt{V_t}\,dW_t^S,\\
V_t
&=V_0+
\frac{1}{\Gamma(H+\tfrac12)}
\int_0^t(t-s)^{H-\frac12}\kappa(\theta-V_s)\,ds\\
&\quad+
\frac{1}{\Gamma(H+\tfrac12)}
\int_0^t(t-s)^{H-\frac12}\xi\sqrt{V_s}\,dW_s^V,\\
d\langle W^V,W^S\rangle_t&=\rho_{V,S}\,dt.
\end{aligned}
$$

- \(V_0\) : variance initiale ; \(\theta\) : niveau long terme de variance.
- \(\kappa\) : coefficient de retour à la moyenne ; \(\xi\) : vol-of-vol.
- \(H\in(0,\tfrac12]\) : exposant constant ; \(\rho_{V,S}\) : corrélation variance–sous-jacent.

La mémoire de la variance est gouvernée par un même exposant \(H\). À courte maturité, le skew rough est associé à une loi de puissance de type \(T^{H-1/2}\), et non à une exponentielle unique. Cette asymptotique ne signifie pas que toute sa courbe de skew est une droite exacte en log-log.

Le projet calcule les prix par résolution de Riccati fractionnaire et intégration de Fourier.

### MPCE : Multifractional Process with Correlated Exponent

Le MPCE fait varier l’exposant de mémoire. Il distingue le facteur \(y_t\), l’exposant \(H_t\) et la variance \(V_t=e^{v_t}\).

Les dynamiques continues des facteurs, avant les corrections numériques, sont :

$$
\begin{aligned}
dy_t&=\kappa(\theta-y_t)\,dt+\xi\sqrt{y_t}\,dW_t^y,\\
dH_t&=\lambda(\eta-H_t)\,dt
+\zeta\sqrt{H_t(1-H_t)}\,dW_t^H.
\end{aligned}
$$

Ici, \(\theta\) est la moyenne long terme de \(y\), pas de \(V\). Pour \(H\), \(\lambda\) est le retour à la moyenne, \(\eta\) la moyenne long terme et \(\zeta\) le coefficient de diffusion.

Les corrélations sont définies par :

$$
d\langle W^i,W^j\rangle_t=\rho_{i,j}\,dt,
\qquad i,j\in\{H,y,S\},
$$

avec, dans l’ordre \((H,y,S)\),

$$
R=\begin{pmatrix}
1&\rho_{H,y}&\rho_{H,S}\\
\rho_{H,y}&1&\rho_{y,S}\\
\rho_{H,S}&\rho_{y,S}&1
\end{pmatrix}.
$$

La log-variance est une convolution des variations de \(y\). Posons :

$$
c(h)=\frac{\sqrt{2h\sin(\pi h)\Gamma(2h)}}{\Gamma(h+\tfrac12)},
\qquad
K(t,s)=c(H_s)(t-s)^{H_s-\frac12}.
$$

Pour \(t\geq0\), le système complet s’écrit :

$$
\begin{aligned}
v_t&=\log V_0+A_t
+\int_0^t K(t,s)\kappa(\theta-y_s)\,ds
+\int_0^t K(t,s)\xi\sqrt{y_s}\,dW_s^y,\\[1mm]
V_t&=e^{v_t},\\[1mm]
\frac{dS_t}{S_t}&=(r-q)\,dt+e^{v_t/2}\,dW_t^S.
\end{aligned}
$$

Pour un historique de longueur \(L\), la mémoire antérieure à \(t=0\) est :

$$
\begin{aligned}
A_t={}&\int_{-L}^{0}
\bigl[K(t,s)-K(0,s)\bigr]\kappa(\theta-y_s)\,ds\\
&+\int_{-L}^{0}
\bigl[K(t,s)-K(0,s)\bigr]\xi\sqrt{y_s}\,dW_s^y.
\end{aligned}
$$

La soustraction garantit \(A_0=0\) : \(V_0\) reste la variance initiale du sous-jacent, tandis que les états terminaux de l’historique initialisent \(y\) et \(H\) au début du pricing. Sans historique, \(A_t=0\).

Le noyau utilise \(H_s\), l’exposant au temps de l’innovation, et non \(H_t\). Sa variation modifie donc la pondération des innovations passées. Les corrélations relient cette mémoire aux mouvements du facteur de volatilité et du sous-jacent.

L’objectif est de produire une transition entre régimes de skew avec un seul facteur de volatilité à exposant variable, plutôt qu’avec deux facteurs de volatilité superposés. Le modèle comporte néanmoins un état stochastique supplémentaire, \(H_t\) : « un facteur » ne signifie pas « une seule source d’aléa ». 

Bornes de Hurst dans le projet. La diffusion continue ci-dessus s’annule en \(0\) et \(1\), pas aux bornes configurées. Dans le code, \(H\) est en plus projeté à chaque pas sur \([H_{\min},H_{\max}]\), actuellement \([0{,}03;0{,}40]\). Ces bornes sont une contrainte numérique supplémentaire : elles ne sont pas imposées par la seule équation continue écrite ici. De même, la discrétisation de \(y\) utilise sa partie positive dans les coefficients. Les équations présentées décrivent la dynamique continue sous-jacente ; elles ne constituent pas une preuve de convergence du schéma projeté.

## 3. Grecques et résultats

Les grecques de Rough Heston mesurent la sensibilité du prix actualisé à H, à la vol-of-vol, au retour à la moyenne et au niveau long terme. Elles sont calculées par différences finies avec pricing Fourier, à strike et autres paramètres fixes.

Dans les résultats fournis, les sensibilités aux paramètres de variance sont globalement plus faibles quand l’échéance approche. La sensibilité à H présente un profil différent : elle est plus importante à l’ATM à courte maturité. Ces observations ne constituent pas des lois de monotonie valables à tous les strikes.

La sensibilité à la vol-of-vol est distincte du volga : elle mesure une dérivée première par rapport au coefficient de diffusion de la variance, tandis que le volga mesure une dérivée seconde par rapport à un niveau de volatilité. Elle peut être négative dans les simulations, notamment à l’ATM : modifier la vol-of-vol transforme la distribution future, sans relever uniformément la volatilité.

Les surfaces et les coupes montrent les sourires ; la structure à terme du skew permet de comparer leur évolution en maturité. L’ajustement en log-log compare une pente unique à deux pentes, donc des régimes de loi de puissance, et non deux exponentielles. Les deux axes des coupes facilitent la comparaison des formes, pas des hauteurs.

## 4. Utilisation

`presentation.ipynb` regroupe les paramètres, le payoff et les graphiques. `rough_heston.py` contient le pricing Fourier et les grecques ; `mpce.py` la simulation multifractionnaire ; `visualisation.py` la présentation et la sauvegarde.

Le bloc Rough Heston peut être exécuté seul. L’exécution complète calcule les deux modèles et remplace les résultats du précédent essai réussi. Les équations continues ci-dessus explicitent les modèles ; les contrôles numériques sont regroupés dans `verification_livraison.json`.
