class load_label(Dataset):
  def __init__(self,csv_file,image_dir,label_dir,S=7,B=1,C=20,transform = None):
    self.path_and_name_image = pd.read_csv(csv_file)
    self.label_dir = label_dir
    self.image_dir = image_dir
    self.len = self.path_and_name_image.iloc[:,0].values.shape[0]
    self.S =  S
    self.B = B
    self.C = C
    self.transform = transform 
  def __getitem__(self,index):
    image_name,label_name = self.path_and_name_image.iloc[index,:]
    image_path,label_path = os.path.join(self.image_dir,image_name),os.path.join(self.label_dir,label_name)
    image = Image.open(image_path)
    image = np.array(image)
    boxes = []
    with open(label_path,'r') as f:
      for line in f.readlines():
        box = [float(i)  for i in line.strip().split()]
        box[0] = int(box[0])
        boxes.append(box)
    target_label = torch.zeros((self.S,self.S,5+self.C))
    image = self.transform(image)
    for label in boxes:
      type_class,x,y,w,h = label
      j,i = int(self.S*x),int(y*self.S)
      w_annotation,h_annotation = w*self.S,h*self.S
      x_annotation,y_annotation = self.S*x -j,self.S*y-i
      if(target_label[i,j,20]==0):
        target_label[i,j,type_class-1] = 1
        target_label[i,j,20:25]= torch.tensor([1,x_annotation,y_annotation,w_annotation,h_annotation])
    return image,target_label
  def __len__(self):
    return  self.len
