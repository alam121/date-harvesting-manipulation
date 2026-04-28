#include "rviz_ur10e_panel/gripper_panel.hpp"

#include <QPainter>
#include <QPainterPath>
#include <QLinearGradient>
#include <QVBoxLayout>
#include <QFont>
#include <QSizePolicy>
#include <cstring>

namespace rviz_ur10e_panel
{

// ── palette ──────────────────────────────────────────────────────────────────
static const QColor BG      {16, 16, 22};
static const QColor BORDER  {40, 40, 55};
static const QColor DIM     {55, 55, 70};
static const QColor TXTLO   {100,100,120};
static const QColor TXTHI   {220,220,235};

// finger contact colours
static QColor fgBase(float f){
  if(f<0.5f) return {38,38,52};
  if(f<2.f)  return {18,90,58};
  if(f<4.f)  return {130,78,0};
  return             {150,30,30};
}
static QColor fgAccent(float f){
  if(f<0.5f) return {55,55,70};
  if(f<2.f)  return {40,200,110};
  if(f<4.f)  return {255,170,20};
  return             {255,70,70};
}

// ── GripperWidget ────────────────────────────────────────────────────────────

GripperWidget::GripperWidget(QWidget * p) : QWidget(p)
{
  setMinimumSize(80, 100);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void GripperWidget::setForces(float l, float c, float r)
{ forces_[0]=l; forces_[1]=c; forces_[2]=r; update(); }

void GripperWidget::setPhase(const std::string & ph)
{
  phase_  = ph;
  closed_ = ph=="FINAL"||ph=="GRASPING"||ph=="REVERSING"||ph=="DROPOFF";
  update();
}

void GripperWidget::setClosureData(bool se, int fc, int ts, int cs)
{ stopped_early_=se; first_contact_=fc; total_steps_=ts; closure_step_=cs; update(); }

void GripperWidget::paintEvent(QPaintEvent *)
{
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing);
  p.fillRect(rect(), BG);

  const qreal W=width(), H=height();
  const qreal M=6;
  const qreal usW = W-2*M;

  // ── closure progress bar (top) ───────────────────────────────────────────
  const qreal pbH=7, pbY=M, pbR=3;
  float prog = (total_steps_>0 && closure_step_>0)
               ? float(closure_step_)/float(total_steps_) : 0.f;
  // background track
  QPainterPath trk; trk.addRoundedRect(M, pbY, usW, pbH, pbR, pbR);
  p.fillPath(trk, BORDER);
  // fill
  if(prog > 0.f){
    QColor fc = stopped_early_ ? QColor(40,200,110) : QColor(200,80,50);
    QPainterPath fill; fill.addRoundedRect(M, pbY, usW*prog, pbH, pbR, pbR);
    p.fillPath(fill, fc);
  }

  // ── palm ─────────────────────────────────────────────────────────────────
  const qreal palmH=14, palmY=pbY+pbH+4;
  QColor palmC = closed_ ? QColor(40,70,130) : QColor(35,35,50);
  QPainterPath palm; palm.addRoundedRect(M, palmY, usW, palmH, 4, 4);
  p.fillPath(palm, palmC);
  p.setPen(BORDER); p.drawPath(palm);
  p.setPen(closed_ ? QColor(140,180,255) : TXTLO);
  QFont pf; pf.setPointSize(6); pf.setBold(true); p.setFont(pf);
  p.drawText(QRectF(M,palmY,usW,palmH), Qt::AlignCenter,
             closed_ ? "CLOSED" : "OPEN");

  // ── three fingers ─────────────────────────────────────────────────────────
  const qreal gap   = usW*0.04;
  const qreal fW    = (usW - 2*gap) / 3.0;
  const qreal fY    = palmY + palmH + 3;
  const qreal fH    = H - fY - M;
  const qreal fx[3] = { M, M+fW+gap, M+2*(fW+gap) };
  const char* nm[3] = {"L","C","R"};

  for(int i=0; i<3; ++i){
    float f = forces_[i];
    QColor base   = fgBase(f);
    QColor accent = fgAccent(f);
    bool   hit    = f >= 0.5f;

    // body
    QLinearGradient g(fx[i], fY, fx[i]+fW, fY);
    g.setColorAt(0, base.lighter(115));
    g.setColorAt(1, base);
    QPainterPath fp; fp.addRoundedRect(fx[i], fY, fW, fH, 4, 4);
    p.fillPath(fp, g);
    p.setPen(hit ? QPen(accent,1.2) : QPen(BORDER,0.8));
    p.drawPath(fp);

    // tip dot
    qreal dotR = fW*0.28;
    qreal dCX  = fx[i]+fW/2, dCY = fY+fH-dotR-5;
    if(hit){
      p.setPen(Qt::NoPen);
      p.setBrush(QColor(accent.red(),accent.green(),accent.blue(),50));
      p.drawEllipse(QRectF(dCX-dotR*1.7,dCY-dotR*1.7,dotR*3.4,dotR*3.4));
      p.setBrush(accent);
    } else {
      p.setPen(Qt::NoPen);
      p.setBrush(DIM);
    }
    p.drawEllipse(QRectF(dCX-dotR,dCY-dotR,dotR*2,dotR*2));

    // force value
    p.setPen(hit ? TXTHI : TXTLO);
    QFont vf; vf.setPointSize(5); p.setFont(vf);
    p.drawText(QRectF(fx[i], fY+fH*0.38, fW, 11),
               Qt::AlignCenter,
               QString::number(double(f),'f',1)+"N");

    // letter
    p.setPen(TXTLO);
    QFont lf; lf.setPointSize(5); p.setFont(lf);
    p.drawText(QRectF(fx[i], fY+2, fW, 10), Qt::AlignCenter, nm[i]);
  }

  // ── contact quality indicator (bottom-right of progress bar) ─────────────
  if(first_contact_ >= 0 && total_steps_ > 0){
    float ratio = float(first_contact_)/float(total_steps_);
    // pill: earlier contact = greener
    QColor qc = (ratio < 0.5f) ? QColor(40,200,100)
              : (ratio < 0.8f) ? QColor(200,150,30)
              :                  QColor(200,60,60);
    p.setPen(Qt::NoPen); p.setBrush(qc);
    p.drawRoundedRect(QRectF(W-M-22, pbY, 22, pbH), 2, 2);
    p.setPen(Qt::white);
    QFont qf; qf.setPointSize(4); p.setFont(qf);
    p.drawText(QRectF(W-M-22, pbY, 22, pbH), Qt::AlignCenter,
               stopped_early_ ? "EARLY" : "FULL");
  }
}

// ── GripperPanel ─────────────────────────────────────────────────────────────

GripperPanel::GripperPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  setStyleSheet("background:#101016; color:#dcdce8;");

  auto * root = new QVBoxLayout(this);
  root->setContentsMargins(5,5,5,5);
  root->setSpacing(3);

  // state badge
  state_lbl_ = new QLabel("○  OPEN");
  state_lbl_->setAlignment(Qt::AlignCenter);
  state_lbl_->setFixedHeight(20);
  state_lbl_->setStyleSheet(
    "background:#1e1e2a; color:#666688; border-radius:4px;"
    "font-size:8pt; font-weight:bold; letter-spacing:1px;");
  root->addWidget(state_lbl_);

  // painted widget — stretch factor 1 so it fills all remaining space
  gw_ = new GripperWidget(this);
  root->addWidget(gw_, 1);

  // quality line
  quality_lbl_ = new QLabel("");
  quality_lbl_->setAlignment(Qt::AlignCenter);
  quality_lbl_->setFixedHeight(14);
  quality_lbl_->setStyleSheet("color:#555566; font-size:6pt;");
  root->addWidget(quality_lbl_);

  timer_ = new QTimer(this);
  connect(timer_, &QTimer::timeout, this, &GripperPanel::updateDisplay);
}

GripperPanel::~GripperPanel() = default;

void GripperPanel::onInitialize()
{
  setupRos();
  timer_->start(100);
}

// helper: extract int field from JSON string
static int extractInt(const std::string & d, const char * key, int def=0){
  auto pos = d.find(key);
  if(pos==std::string::npos) return def;
  auto colon = d.find(':', pos);
  if(colon==std::string::npos) return def;
  auto vs = colon+1;
  while(vs<d.size() && (d[vs]==' '||d[vs]=='\t')) vs++;
  return std::stoi(d.substr(vs));
}
static bool extractBool(const std::string & d, const char * key){
  auto pos = d.find(key);
  if(pos==std::string::npos) return false;
  auto colon = d.find(':', pos);
  if(colon==std::string::npos) return false;
  auto vs = colon+1;
  while(vs<d.size() && (d[vs]==' '||d[vs]=='\t')) vs++;
  return d.substr(vs,4)=="true";
}
static std::string extractStr(const std::string & d, const char * key){
  auto pos = d.find(key);
  if(pos==std::string::npos) return "";
  auto q1 = d.find('"', pos+std::strlen(key));
  auto q2 = (q1!=std::string::npos) ? d.find('"',q1+1) : std::string::npos;
  if(q2==std::string::npos) return "";
  return d.substr(q1+1, q2-q1-1);
}

void GripperPanel::setupRos()
{
  node_ = std::make_shared<rclcpp::Node>("gripper_panel_node");

  force_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
    "/gripper/force", 10,
    [this](std_msgs::msg::Float32MultiArray::SharedPtr msg){
      std::lock_guard<std::mutex> lk(mutex_);
      for(int i=0;i<3&&i<(int)msg->data.size();++i) forces_[i]=msg->data[i];
    });

  info_sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/goal_info", 10,
    [this](std_msgs::msg::String::SharedPtr msg){
      const std::string & d = msg->data;
      std::string ph = extractStr(d, "\"motion_phase\"");
      bool   se = extractBool(d, "\"gripper_stopped_early\"");
      int    fc = extractInt (d, "\"gripper_first_contact\"", -1);
      int    ts = extractInt (d, "\"gripper_steps\"",          10);
      int    cs = extractInt (d, "\"gripper_closure_step\"",   -1);
      std::lock_guard<std::mutex> lk(mutex_);
      phase_=ph; stopped_early_=se; first_contact_=fc;
      total_steps_=ts; closure_step_=cs;
    });
}

void GripperPanel::updateDisplay()
{
  rclcpp::spin_some(node_);

  float f0,f1,f2; std::string ph; bool se; int fc,ts,cs;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    f0=forces_[0]; f1=forces_[1]; f2=forces_[2];
    ph=phase_; se=stopped_early_; fc=first_contact_; ts=total_steps_; cs=closure_step_;
  }

  gw_->setForces(f0,f1,f2);
  gw_->setPhase(ph);
  gw_->setClosureData(se,fc,ts,cs);

  // state badge
  bool contact = (f0>0.5f||f1>0.5f||f2>0.5f);
  QString txt; QString sty;
  const QString B = "border-radius:4px;font-size:8pt;font-weight:bold;letter-spacing:1px;";
  if     (ph=="FINAL"||ph=="GRASPING")  { txt="⬤  GRASPING"; sty=B+"background:#0b3d18;color:#3ddd77;"; }
  else if(ph=="REVERSING"&&contact)     { txt="⬤  HOLDING";  sty=B+"background:#0b2450;color:#4499ff;"; }
  else if(ph=="DROPOFF")                { txt="◎  RELEASING";sty=B+"background:#3d2600;color:#ffaa33;"; }
  else if(ph=="APPROACH")               { txt="→  APPROACH"; sty=B+"background:#12123a;color:#7777ff;"; }
  else if(contact)                      { txt="⬤  CONTACT";  sty=B+"background:#333300;color:#dddd22;"; }
  else                                  { txt="○  OPEN";     sty=B+"background:#1e1e2a;color:#666688;"; }
  state_lbl_->setText(txt);
  state_lbl_->setStyleSheet(sty);

  // quality line
  if(fc>=0 && ts>0){
    float ratio = float(fc)/float(ts);
    QString qstr = se
      ? QString("contact @ step %1/%2 — PROPER").arg(fc).arg(ts)
      : QString("fully closed (%1/%2) — NO CONTACT").arg(cs<0?ts:cs).arg(ts);
    QString qcol = se ? (ratio<0.5f ? "#3ddd77" : "#ffaa33") : "#dd4444";
    quality_lbl_->setStyleSheet(QString("color:%1;font-size:6pt;").arg(qcol));
    quality_lbl_->setText(qstr);
  } else {
    quality_lbl_->setText("");
  }
}

}  // namespace rviz_ur10e_panel

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz_ur10e_panel::GripperPanel, rviz_common::Panel)
