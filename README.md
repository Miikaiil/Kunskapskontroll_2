# Cicek Insight | Project MNIST 🚀

Detta repository innehåller mitt arbete för **Kunskapskontroll 2** i kursen Machine Learning. Projektet omfattar både en teoretisk del med fokus på Python och ML-koncept, samt en praktisk del där jag modellerat MNIST-datasetet och byggt en interaktiv Streamlit-applikation.

## 🧠 Resan från Notebook till Produktion

Arbetet inleddes med att utforska MNIST-datan, som består av 70 000 gråskalebilder av handskrivna siffror. Genom hela projektet har jag dokumenterat tidsåtgång och effektivitet för att säkerställa ett professionellt arbetsflöde.

### Tekniska lärdomar & "Trial and Error"
> "Jag har prövat att testa flera olika modeller och metoder, flera olika parametrar och hyperparametrar. Jag körde boken rakt av initialt men hamnade snabbt i en engagerad jakt på högsta möjliga Accuracy efter diskussioner i Discord-chatten."

* **Hyperparameter-optimering:** Genom att analysera min **Confusion Matrix** lyckades jag pressa en modell till hela 99,8% noggrannhet genom att specifikt träna på de siffror som predikterades fel.
* **Verklighetschecken:** När jag byggde min Streamlit-app insåg jag att hög noggrannhet i en Notebook inte alltid översätts till en bra användarupplevelse på "ny data".
* **Preprocessing är nyckeln:** Det var först när jag implementerade **smart centrering** som resultaten i appen verkligen lyfte. Jag experimenterade även med **HOG (Histogram of Oriented Gradients)** för att extrahera särdrag.
* **Effektivitet vs. Precision:** En viktig insikt var att modeller som tog några minuter att träna ofta presterade lika bra i praktiken som de som tog timmar att köra för en marginell procentuell vinst.



## 🛠 Systemarkitektur
Den slutgiltiga applikationen använder en hybridlösning för att uppfylla kraven på att prediktera ny data:

* **Modell:** En `VotingClassifier` som kombinerar **Random Forest** och **SVC** (Support Vector Classifier).
* **Multi-Digit Scanning:** Implementering av **OpenCV** för att segmentera ritytan och identifiera flera siffror i en sekvens.
* **Feature Heatmap:** En visuell funktion som visar modellens "neurala fokus" (viktade pixlar) i en anpassad Copper-palett.



## 📈 Personliga Reflektioner
Detta projekt har inneburit många "Ctrl+A" och nystarter i både presentation och kod. Det roligaste med resan är att jag redan har börjat tillämpa detta tänk i mitt arbetsliv. 

Jag har nyligen klivit in i en roll som **Lean-ledare med fokus på digitalisering**, och metodiken från denna kurs är något jag applicerar i min yrkesroll redan idag. Jag längtar till nästa kurs för att få fördjupa mig ännu mer!



## 📁 Inlämningsinnehåll
* `notebook.ipynb`: Mitt kompletta ML-flöde och modellträning.
* `app.py`: Källkod för Streamlit-applikationen.
* `Teoretiska_frågor.pdf`: Svar på de 19 teoretiska frågorna.
* `mnist_voting_model_final.pkl`: Den färdigtränade modellen (via Git LFS).

---
*Tack för en inspirerande kurs och en kanonbra bok!*

*Ville pröva att köra hela vägen med att skapa en API, så resultatet är:*
*https://cicekinsight.com (Vet inte hur länge den ligger kvar, men iaf rättningen ut, det är min app lanserad i HTML med API via Huggingface <img width="25" height="20" alt="image" src="https://github.com/user-attachments/assets/74d24f22-7c3a-43ec-bd61-911a8e4fa846" />* 
