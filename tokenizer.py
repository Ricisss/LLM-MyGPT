from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# 1. Initialize a blank BPE model
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel()

tokenizer.decoder = decoders.ByteLevel()

# 2. Define a trainer with your target vocabulary size (e.g., 16000)
trainer = BpeTrainer(vocab_size=16000, 
                        special_tokens=["<unk>", "#TITLE#", "#CONTENT#", "#END#"],
                        initial_alphabet=ByteLevel.alphabet()
                        )

# 3. Train the tokenizer on your dataset file
tokenizer.train(files=["dataset.txt"], trainer=trainer)

# decoding 
output = tokenizer.encode("Ahoj, novinky!")
print("Encoded:", output.ids)
print("Decoded:", tokenizer.decode(output.ids)) 
tokenizer.save("tokenizer.json")