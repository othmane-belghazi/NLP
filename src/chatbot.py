flowchart TD
    Start([▶ AutoMoteurTerme.process_task]):::entry

    subgraph MOTEUR["MoteurTerme.py — Orchestrateur"]
        direction LR
        S1[Clean cache] --> S2[get_sources<br/>df_terme] --> S3[ScoringTerme] --> S4[JOIN BV ⨝ Scoring<br/>+ DateTerme] --> S5[ELR<br/>+ POL_PrimePure] --> S6[P3 MA + MCC<br/>unionByName]
    end

    subgraph PREP["preparation/ — Mise en forme & scope"]
        direction LR
        PP["Preprocessing.process<br/>fill → missing → array → mapping →<br/>cast → diff_dates → init → price_test → r0"]
        ST[ScopeTerme<br/>in/out scope]
        PP --> ST
    end

    subgraph ENGINE["MajorationEngine.process — Cœur métier"]
        direction TB

        subgraph CLAIMS["① Claims & LCA"]
            direction LR
            CHS["Claims_hsMED<br/>transpose → duree_FD →<br/>compute → aggregate"]
            CMED["Claims_MED<br/>transpose → compute → aggregate"]
            CCO["CreditCo<br/>transpose → compute → aggregate"]
            JOIN1["JOIN df ⨝<br/>claims/med/lca"]
            CYC[Claims_ycMED]
            DROP[drop CLA/CC<br/>cols >6/4]
            CHS --> JOIN1
            CMED --> JOIN1
            CCO --> JOIN1
            JOIN1 --> CYC --> DROP
        end

        subgraph ROW1[" "]
            direction LR
            SCOR["② MajoSin2<br/>MajoSinGar_F0 → MajoSinUT"]
            CFT["③ CotisFraisTaxesELR<br/>taxes → Frais → Cotis →<br/>Cotis_hsMajo → CotisP3 → ELR"]
            SEG["④ Segmentation<br/>Sinistralite → Mesures →<br/>Protection → MiniEuros"]
            CHURN["⑤ ChurnModel<br/>coeffModel → ProbaModel →<br/>PriceSensitivity"]
            SCOR --> CFT --> SEG --> CHURN
        end

        subgraph MAJ_BLOCK["⑥ ModuleMajoration"]
            direction LR
            subgraph MAJ_PIPE[" "]
                direction LR
                M1[ajust_ELR] --> M2[CCAS] --> M3[EO] --> M4[Brok] --> M5[Colla] --> M6[DmD] --> MF[(MajoFinale)] --> M7[cotis_F0]
            end
            subgraph MFIN["MajoFinale — 28 règles séquentielles"]
                direction LR
                R1["Scoring<br/>scoringSinistre →<br/>scoringMED"] --> R2["CCAS<br/>carburant → formule →<br/>marque → modeachat"] --> R3["Tolérance & butoirs<br/>tolerancesinistre → BDG_inf200 →<br/>ClientsRecents → CDB"]
                R3 --> R4["Contrats<br/>Premature → Fragile → Remp2M →<br/>Rentables → SansAnt/EOM/MA →<br/>Jeune_T3_CUT"] --> R5["RIVierge & CRM<br/>RIVierge_hsBLD → BaisseCRM1 →<br/>RIVierge_ycBLD"] --> R6["VEH<br/>haute_gamme"]
                R6 --> R7["Butoirs AGIRA<br/>agira →<br/>sinistreFraude"] --> R8["Butoirs surprotégés<br/>MultiEquipes → CollabAgents →<br/>SuperDetenteur_sup6"] --> R9["Finalisations<br/>extremes_pente →<br/>MonCampingCar → mini"]
            end
            MF -.contient.-> MFIN
        end

        subgraph POST["⑦ Post-traitements"]
            direction LR
            PT[PriceTest] --> AJ[Ajustement<br/>parent-child] --> CT[ComputeCT] --> CPB[CotisPlancher<br/>Butoir] --> BM[Bulle<br/>Marketing] --> FD[Franchise<br/>Degressive] --> SG[CodeSGMT] --> PE[Patch_errors] --> AX[AxaPac]
        end

        CLAIMS --> ROW1 --> MAJ_BLOCK --> POST
    end

    subgraph OUT["MoteurTerme — Sorties"]
        direction LR
        O1[MarquageISA_F0<br/>+ tech_version] --> O2[write<br/>auto_moteur_terme] --> O3[write<br/>auto_moteur_terme_ko] --> O4{write_mainframe?}
        O4 -- oui --> O5[generate_mainframe<br/>contrats_auto_termostat]
        O4 -- non --> Done([■ Fin])
        O5 --> Done
    end

    Start --> MOTEUR --> PREP --> ENGINE --> OUT

    classDef entry fill:#1f6feb,color:#fff,stroke:#0b3d91,stroke-width:2px;