# ====================================================================
# Script PowerShell: Automatizza combinazione DC per tutti gli output
# ====================================================================
# Questo script può essere eseguito in QUALSIASI MOMENTO dopo che
# client.py ha generato i file DC_Predicate_Stats.json
# NON servono server/client attivi!
# ====================================================================

Write-Host "`n AUTOMAZIONE COMBINAZIONE DENIAL CONSTRAINTS`n" -ForegroundColor Cyan

# === CONFIGURAZIONE ===
$noise_values = @(0, 2, 4, 8)
$similarity_values = @(0.5, 0.75, 0.9, 1.0)
$modalities = @("All_Sensitive", "All_NOT_Sensitive", "Manual")  # Aggiungi/rimuovi modalità
$dataset = "hepatitis"

# Parametri per il combinatore
$min_support = 0.95              # Support minimo predicati atomici
$min_combined_support = 0.80     # Support minimo DC combinate
$max_combo_size = 3              # Dimensione massima combinazioni

# === CONTATORI ===
$total_processed = 0
$total_success = 0
$total_skipped = 0
$total_errors = 0

Write-Host " Configurazione:" -ForegroundColor Yellow
Write-Host "   - Dataset: $dataset"
Write-Host "   - Modalità: $($modalities -join ', ')"
Write-Host "   - Noise values: $($noise_values -join ', ')"
Write-Host "   - Similarity values: $($similarity_values -join ', ')"
Write-Host "   - Min support: $min_support"
Write-Host "   - Min combined support: $min_combined_support"
Write-Host "   - Max combo size: $max_combo_size"
Write-Host ""

# === ELABORAZIONE ===
foreach ($modality in $modalities) {
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " Modalità: $modality" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    
    foreach ($noise in $noise_values) {
        foreach ($similarity in $similarity_values) {
            $output_dir = "Test/$dataset/$modality/output_N${noise}_S${similarity}"
            $stats_file = "$output_dir/DC_Predicate_Stats.json"
            
            $total_processed++
            
            if (Test-Path $stats_file) {
                Write-Host "`n📊 [$total_processed] Elaborazione: output_N${noise}_S${similarity}" -ForegroundColor Green
                
                try {
                    # Esegui combine_dcs.py
                    python combine_dcs.py $stats_file $output_dir $min_support $min_combined_support $max_combo_size
                    
                    if ($LASTEXITCODE -eq 0) {
                        $total_success++
                        Write-Host "    Completato con successo" -ForegroundColor Green
                    } else {
                        $total_errors++
                        Write-Host "   Errore durante l'elaborazione" -ForegroundColor Red
                    }
                }
                catch {
                    $total_errors++
                    Write-Host "    Eccezione: $_" -ForegroundColor Red
                }
            }
            else {
                $total_skipped++
                Write-Host "`n [$total_processed] Saltato: output_N${noise}_S${similarity} (file non trovato)" -ForegroundColor Yellow
            }
        }
    }
}

# === RIEPILOGO FINALE ===
Write-Host "`n═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "RIEPILOGO FINALE" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   Totale directory elaborate: $total_processed"
Write-Host "    Successi: $total_success" -ForegroundColor Green
Write-Host "     Saltate: $total_skipped" -ForegroundColor Yellow
Write-Host "    Errori: $total_errors" -ForegroundColor Red
Write-Host ""

if ($total_success -gt 0) {
    Write-Host " Completato! Controlla i file DCs_combined.txt in ogni directory" -ForegroundColor Green
}

if ($total_errors -gt 0) {
    Write-Host "  Alcuni output hanno generato errori. Verifica i log sopra." -ForegroundColor Yellow
}

Write-Host ""
