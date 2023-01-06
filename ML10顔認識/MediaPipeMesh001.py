import numpy
import cv2
import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

triangles = []

def imread(frame):
	if frame is None:
		return None
	image = frame.numpyArray(delayed=True) # フレームをnumpy array化
	image *= 255
	image = image.astype('uint8')
	image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
	# image = cv2.flip(image, 0)
	return image

# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
	page = scriptOp.appendCustomPage('source')
	p = page.appendTOP('Image', label='Video image')
	m = page.appendDAT('Polygons', label='Polygons data')
	b = page.appendPulse('Load', label='Load data')
	''' x = page.appendFloat('Valuex', label='Value X')
	y = page.appendFloat('Valuey', label='Value Y')
	z = page.appendFloat('Valuez', label='Value Z') '''
	return

# called whenever custom pulse parameter is pushed
def onPulse(par):
	if par.name == 'Load':
		op = par.owner

		pols = op.par.Polygons.eval()
		for i in range(pols.numRows):
			vertices = [int(pols[i, 0]), int(pols[i, 1]), int(pols[i, 2])]
			triangles.append(vertices)
	return

def onCook(scriptOp):
	scriptOp.clear()

	# 次のフレームを読み込む
	image = imread(frame = scriptOp.par.Image.eval()) # フレーム入力
	if image is None: # 読み込み失敗
		return
	
	# 顔を検出
	results = face_mesh.process(image)
	if results.multi_face_landmarks is None: # 検出失敗
		return

	# Mesh 作成
	firstface = results.multi_face_landmarks[0] # 最初の顔（しかないはず）
	asp = image.shape[0]/image.shape[1]
	for pt in firstface.landmark:
		p = scriptOp.appendPoint()
		p.P = (pt.x,(1 - pt.y)*asp,pt.z)
	'''p = scriptOp.points[6]
	scriptOp.par.Valuex = p.x
	scriptOp.par.Valuey = p.y
	scriptOp.par.Valuez = p.z'''

	for poly in triangles:
		pp = scriptOp.appendPoly(3, closed=True, addPoints=False)
		pp[0].point = scriptOp.points[poly[0]]
		pp[1].point = scriptOp.points[poly[1]]
		pp[2].point = scriptOp.points[poly[2]]

	return
