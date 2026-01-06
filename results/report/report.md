# Micro Bonus 2026 - Soft Error Analysis on DNNs


## Εισαγωγή - Κίνητρο 


## Σύγκριση μεταξύ διαφορετικών data types (fp32, fp16, int8, int4)

Τα μοντέλα μας δοκιμάστηκαν με 2 ειδών data types: floating point (των 32 και 16 bit) και integers (των 8 και 4 bit). 

Σε μια πρώτη ανάλυση των αποτελεσμάτων, παρατηρούμε το robustness των fixed-point/integers απέναντι στα εισαγώμενα σφάλματα. Πράγματι, παρά την αύξηση του BER (έως και 1e-3) διατηρούν εν γένει υψηλή ακρίβεια. 


Σε όλα τα μοντέλα (ResNet18 και ResNet50) και σε όλα τα datasets, το πιο συνεπές μοτίβο είναι η άμεση αποτυχία των μορφών κινητής υποδιαστολής (FP16/FP32) όταν όλα τα bits είναι επιρρεπή σε αλλοίωση.

Τάση: Ακόμα και σε εξαιρετικά χαμηλά BER ($10^{-7}$ ή $10^{-6}$), οι γραμμές “all bits” για FP καταρρέουν άμεσα στην τυχαία ακρίβεια (random guessing).

Συμπέρασμα: Σε πραγματική χρήση χωρίς ECC, τα μοντέλα κινητής υποδιαστολής είναι πρακτικά μη χρησιμοποιήσιμα αν υπάρχει οποιοσδήποτε κίνδυνος bit-flips στη μνήμη. Η προστασία των bits του εκθέτη δεν είναι απλώς βελτιστοποίηση· είναι απαραίτητη για τη σταθερότητα του μοντέλου.

2. Διαχωρισμός Mantissa vs. All bits

Η ανάλυση της γραμμής στη διάκριση “mantissa” έναντι “all” αποκαλύπτει τη λογική της αποτυχίας:

FP16/FP32 Mantissa (μόνο δεκαδικό μέρος): Οι αριθμοί παραμένουν πολύ σταθεροί—παρόμοια με τους ακέραιους. Αλλαγές μόνο στο mantissa αλλάζουν την ακρίβεια, όχι τη μεγέθυνση των τιμών.

FP16/FP32 All bits: Μόλις συμπεριληφθούν τα bits του εκθέτη, η ακρίβεια καταρρέει ακαριαία.

3. Η περίπτωση των Int4

Παρατήρηση: Το int4 | all έχει χαμηλότερη ακρίβεια από το int8 | all.

Λόγος: Το int4 ξεκινά με χαμηλότερη ακρίβεια λόγω quantization error.

Σημείωση: Αν και ξεκινά χαμηλότερα, η καμπύλη του δεν φθίνει ταχύτερα από το int8· απλώς είναι από τη φύση της λιγότερο ακριβής.

4. Δυσκολία dataset vs. Αντοχή σε σφάλματα

Σύγκριση ResNet18 σε CIFAR-10 vs CIFAR-100 δείχνει ότι τα πιο σύνθετα tasks είναι πιο ευάλωτα:

Στο CIFAR-10 (10 κλάσεις), το ResNet18 διατηρεί ικανοποιητική ακρίβεια με int4 | all ακόμα και σε BER $10^{-3}$.

Στο CIFAR-100 (100 κλάσεις), η ίδια διαμόρφωση ξεκινά με χαμηλότερη ακρίβεια και φθίνει ταχύτερα.

Συμπέρασμα: Όσο μεγαλύτερο και πιο πολύπλοκο το dataset (π.χ., ImageNet), τόσο μικραίνει το “safety margin” για quantization και bit-errors. Ένα μοντέλο που φαίνεται fault-tolerant σε απλή εργασία μπορεί να είναι πολύ ευάλωτο σε σύνθετες εργασίες.

5. Βάθος μοντέλου και “Στατιστική Αθροιστική” Επίδραση

Σύγκριση ResNet18 vs ResNet50 στο CIFAR-100:

Το ResNet50 φαίνεται ελαφρώς πιο ανθεκτικό σε υψηλό BER για int8 βάρη.

Στο ResNet50, η γραμμή int8 | all παραμένει πιο “flat” για μεγαλύτερο διάστημα σε σχέση με το ResNet18.

Συμπέρασμα: Τα βαθύτερα μοντέλα μπορεί να έχουν στατιστική αθροιστική επίδραση. Με περισσότερες παραμέτρους, η επίδραση μερικών corrupted weights “διαχέεται” ανάμεσα στις πολλές υγιείς διαδρομές του δικτύου. Μεγαλύτερα μοντέλα μπορεί να είναι πιο ασφαλή σε περιβάλλοντα υψηλής ακτινοβολίας ή χαμηλής τάσης.

![alt text](accuracy_corrupted_vs_ber_comparison.png)
![alt text](resnet18_cifar10_accuracy_vs_ber.png)
![alt text](resnet18_cifar100_accuracy_vs_ber.png)
![alt text](resnet50_cifar100_accuracy_vs_ber.png)


## Η περίπτωση των stuck-at errors

- Υλοποίηση σε κώδικα
- Εξαγωγη ομοιων διαγραμμάτων 
- Σχόλια

## Ανάλυση ευαισθησίας δικτύων σε σφάλματα

- Ουσιαστικά ίδιο με το πρώτο 


## Η χρήση Error Correcting Codes (ECC)

- Κάνω απαραίτητες διορθώσεις
- Ξανατρέχω με ECC
- Διαγράμματα 
- Σχόλια 


Single Error Correction, Double Error Detection
Hamming Code by itself can correct 1-bit errors, but will become confused when there are 2-bit errors. Single Error Correction, Double Error Detection (SECDED) extends Hamming Code with an additional parity bit (ie the first dark green parity bit). This bit tracks the parity of the whole message, so that we can detect 2-bit errors (without being able to correct them). With this additional parity bit, the overall parity of the message should be even. If there is a 1-bit error, the regular parity bits will detect an error and the overall parity of the message is 1; we can assume there is a 1-bit error. If there is a 2-bit error, the regular parity bits will detect an error BUT the overall parity of the message is 0; we have detected a double error.
Efficiency and Limitations
Of course, by having some parity bits, not all bits can be used to transmit data. In this case, we need 5 parity bits to track 11 bits of data for an overall efficiency of 68.75%. Longer messages loosely correlate with higher efficiency. The longer the message, however, the more likely the chance of bit errors, rendering Hamming Code insufficient, since it cannot detect 3 or more errors.

https://medium.com/@ckekula/hamming-code-and-failures-in-semiconductor-main-memory-5f29a129c1e4