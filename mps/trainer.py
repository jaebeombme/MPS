import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

class MRIPulseSequenceClassificationTrainer(object):
    def __init__(
        self,
        learning_rate: float = 1e-3,   # 학습률
        weight_decay:  float = 1e-5,   # L2 정규화 계수
        num_epochs:    int   = 10,     # 전체 학습 에폭 수
        device:        str   = 'cuda' if torch.cuda.is_available() else 'cpu', # 학습 디바이스(GPU/CPU)
        es_patience:   int   = 5,
        output_dir:    str   = ""
    ):
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.num_epochs    = num_epochs
        self.device        = device
        self.es_patience   = es_patience
        self.best_val_loss = float('inf')
        self.patience_cnt  = 0
        self.output_dir    = output_dir

        # 에폭별 손실/정확도 기록용
        self.logs = {
            'train_loss': [],
            'train_acc':  [],
            'valid_loss': [],
            'valid_acc':  [],
        }

    def train_step(self, model, loader, optimizer, criterion, scheduler):
        """
        학습 데이터 전체를 한 번 순회(1에폭)하며
        - 파라미터 업데이트
        - 배치별 손실 및 정확도 집계
        """
        model.train()  # 학습 모드
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
            images = batch['image'].to(self.device)
            labels = batch['label'].long().to(self.device)

            assert labels.dtype == torch.long

            optimizer.zero_grad() 
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    
            scheduler.step()    
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # if batch_idx < 1:
            #     print(f"\n🔹 Batch {batch_idx+1}")
            #     print(f"  ▶ Loss        : {loss.item():.4f}")
            #     print(f"  ▶ Ground Truth: {labels.detach().cpu().numpy().tolist()}")
            #     print(f"  ▶ Prediction  : {preds.detach().cpu().numpy().tolist()}")
            #     print(f"  ▶ logits      :\n{outputs.detach().cpu().numpy()}")

        avg_loss = running_loss / total  
        accuracy = correct / total             
        return avg_loss, accuracy

    def valid_step(self, model, loader, criterion):
        """
        검증 데이터 평가(파라미터 업데이트 없이 손실/정확도 계산)
        """
        model.eval() 
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Valid", leave=False):
                images = batch['image'].to(self.device)
                labels = batch['label'].long().to(self.device)

                assert labels.dtype == torch.long

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, model, train_loader, valid_loader):
        """
        전체 학습 과정을 수행
        - 각 에폭마다 train_step, valid_step 실행
        - 에폭별 손실/정확도 기록
        """
        # 로그 초기화
        self.logs = {
            'train_loss': [],
            'train_acc':  [],
            'valid_loss': [],
            'valid_acc':  [],
        }
        model.to(self.device) 
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.NLLLoss()
        scheduler = CosineAnnealingLR(optimizer, len(train_loader) * self.num_epochs)

        best_acc = 0
        self.final_epoch = 0

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            train_loss, train_acc = self.train_step(model, train_loader, optimizer, criterion, scheduler)
            valid_loss, valid_acc = self.valid_step(model, valid_loader, criterion)

            print(f"  ▶ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  ▶ Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")

            self.logs['train_loss'].append(train_loss)
            self.logs['train_acc'].append(train_acc)
            self.logs['valid_loss'].append(valid_loss)
            self.logs['valid_acc'].append(valid_acc)
            
            if valid_acc > best_acc:
                best_acc = valid_acc    

            if valid_loss < self.best_val_loss:
                self.best_val_loss = valid_loss
                self.patience_cnt = 0
                torch.save(
                    model.state_dict(), 
                    os.path.join(self.output_dir, f'checkpoint_{epoch}.pt')
                )
            else:
                self.patience_cnt += 1
                print(f"  ▶ EarlyStopping counter: {self.patience_cnt}/{self.es_patience}")
                if self.patience_cnt >= self.es_patience:
                    self.final_epoch = epoch
                    print("  ▶ Early stopping triggered.")
                    break

        return self.best_val_loss, best_acc

    def test(self, model, test_loader):
        """
        테스트 데이터로 최종 평가(파라미터 업데이트 없이 손실/정확도 계산)
        """
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = self.valid_step(model, test_loader, criterion)
        print(f"\nTest  ▶ Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
        return test_loss, test_acc

    def plot_logs(self):
        """
        학습/검증 손실 및 정확도 곡선 시각화
        """
        epochs = range(1, self.final_epoch + 1)
        plt.figure(figsize=(12,5))

        # 손실 곡선 그리기
        plt.subplot(1,2,1)
        plt.plot(epochs, self.logs['train_loss'], label='Train Loss')
        plt.plot(epochs, self.logs['valid_loss'], label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        # 정확도 곡선 그리기
        plt.subplot(1,2,2)
        plt.plot(epochs, self.logs['train_acc'], label='Train Acc')
        plt.plot(epochs, self.logs['valid_acc'], label='Valid Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()